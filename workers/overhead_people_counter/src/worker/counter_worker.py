"""
CrowdCount Single-Session Worker Entrypoint

Runs ONE camera's counting loop with no orchestrator, no MediaMTX, no Flask.
Configuration comes entirely from environment variables so the process can run
as a container-per-session under the Upstream processor supervisor (whose
CrowdCountProcessManager writes these CC_* variables into the container env),
or standalone for testing.

The counting behavior is the unmodified production PeopleCounterProcessor —
danger-zone crossing logic, BoT-SORT tracking, V6 model, the 60s
direction_flipped poll, and signed count posting all run exactly as deployed.
This module only wires a single instance to a single stream and manages the
process lifecycle around it.

Environment contract (keep in lock-step with upstream-processor's
app/runtimes/crowdcount_runtime/process_manager.py):

    CC_RTSP_URL          required  Stream source. Upstream passes the session's
                                   internal RTSP URL; a local video file path
                                   also works for smoke tests.
    CC_DEVICE_ID         optional  CrowdCount device id (e.g. "C4804"). Both
                                   credentials empty => PROBE MODE: counting
                                   runs but nothing is posted (the API client
                                   disables itself without credentials).
    CC_DEVICE_PASSWORD   optional  Device password. Setting exactly one of
                                   id/password is a misconfiguration (exit 3).
    CC_API_BASE_URL      optional  Default https://api.crowdcount.tech
    CC_STREAM_NAME       optional  Log prefix / stream name. Defaults to the
                                   device id, else "camera0".
    CC_MODEL_PATH        optional  Override the model weights path (testing).
    CC_FRAME_WIDTH       optional  Informational only (logged); the capture
    CC_FRAME_HEIGHT      optional  reads actual stream properties itself.
    CC_FRAME_FPS         optional

Lifecycle:
    - Posts device "online" to crowdcount-server before the loop and "offline"
      after it (single-shot, best-effort), replacing the old orchestrator's
      status pushes. In service mode a REJECTED online post is fatal (exit 3):
      it means every count post would 401 into the void.
    - Prints CROWDCOUNT_READY on stdout once the processor is initialized —
      the marker the Upstream CrowdCountWorker attaches on.
    - SIGTERM/SIGINT request a cooperative stop, honored during startup as
      well as during the frame loop; total shutdown (stop -> count thread
      grace -> offline post) fits inside Upstream's 8s docker stop grace.

Exit codes:
    0  stop was requested (SIGTERM/SIGINT) or the processor reached ready and
       ended naturally (e.g. stream loss -> supervisor/Docker restarts it)
    1  never reached ready (stream connect or model init failure)
    2  missing CC_RTSP_URL
    3  credentials misconfigured, rejected, or unverifiable (API unreachable)

NOTE: env validation must stay importable/runnable without the CV stack, so
the heavy imports (cv2/ultralytics via the processors package) happen lazily
inside main() after validation.
"""

import os
import sys
import signal
import threading
import time
import logging
import multiprocessing

DEFAULT_API_BASE_URL = 'https://api.crowdcount.tech'

# Stdout marker the Upstream CrowdCountWorker attaches on. Keep in lock-step
# with upstream-processor's app/runtimes/crowdcount_runtime/worker.py.
READY_MARKER = 'CROWDCOUNT_READY'

# Grace given to in-flight fire-and-forget count threads (spawned per crossing
# by the existing processor code) before the process exits.
COUNT_THREAD_GRACE_SECONDS = 1.5

# Timeout for the lifecycle client's online/offline posts. Shorter than the
# processor's own client (5s) so the SIGTERM path fits Upstream's 8s grace.
LIFECYCLE_API_TIMEOUT_SECONDS = 3

logger = logging.getLogger('CounterWorker')


class WorkerConfig:
    """Parsed + validated CC_* environment."""

    def __init__(self, rtsp_url, device_id, password, api_base_url,
                 stream_name, model_path, frame_width, frame_height, frame_fps,
                 service_secret=''):
        self.rtsp_url = rtsp_url
        self.device_id = device_id
        self.password = password
        self.service_secret = service_secret
        self.api_base_url = api_base_url
        self.stream_name = stream_name
        self.model_path = model_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_fps = frame_fps

    @property
    def service_mode(self):
        """Shared service secret + device_id => post on behalf of the device
        without a per-device password (the Upstream worker path)."""
        return bool(self.service_secret and self.device_id)

    @property
    def probe_mode(self):
        """No way to authenticate => counting runs but nothing posts."""
        return not (self.service_mode or (self.device_id and self.password))


def parse_env(environ=None):
    """
    Build a WorkerConfig from the environment.

    Exits the process on invalid configuration (codes 2/3) so a misconfigured
    container fails fast and visibly instead of counting into the void.
    """
    env = os.environ if environ is None else environ

    rtsp_url = env.get('CC_RTSP_URL', '').strip()
    if not rtsp_url:
        print('ERROR: CC_RTSP_URL is required (the stream to count on)',
              file=sys.stderr, flush=True)
        sys.exit(2)

    device_id = env.get('CC_DEVICE_ID', '').strip()
    password = env.get('CC_DEVICE_PASSWORD', '').strip()
    service_secret = env.get('CC_SERVICE_SECRET', '').strip()

    # Service auth needs the target device_id but no per-device password. The
    # id/password "both or neither" rule only applies without a service secret.
    if service_secret:
        if not device_id:
            print('ERROR: CC_SERVICE_SECRET requires CC_DEVICE_ID (the device '
                  'to post on behalf of)', file=sys.stderr, flush=True)
            sys.exit(3)
    elif bool(device_id) != bool(password):
        print('ERROR: CC_DEVICE_ID and CC_DEVICE_PASSWORD must be set together '
              '(set both for per-device counting, neither for probe mode)',
              file=sys.stderr, flush=True)
        sys.exit(3)

    return WorkerConfig(
        rtsp_url=rtsp_url,
        device_id=device_id,
        password=password,
        service_secret=service_secret,
        api_base_url=env.get('CC_API_BASE_URL', '').strip() or DEFAULT_API_BASE_URL,
        stream_name=env.get('CC_STREAM_NAME', '').strip() or device_id or 'camera0',
        model_path=env.get('CC_MODEL_PATH', '').strip(),
        frame_width=env.get('CC_FRAME_WIDTH', ''),
        frame_height=env.get('CC_FRAME_HEIGHT', ''),
        frame_fps=env.get('CC_FRAME_FPS', ''),
    )


def _build_processor(config):
    """Construct the single-session processor (imports the CV stack)."""
    from processors.people_counter_processor import PeopleCounterProcessor

    class SingleSessionPeopleCounter(PeopleCounterProcessor):
        """
        The production people counter, unmodified, plus what a supervised
        single-session worker needs: an attach marker on stdout, an optional
        weights override for testing, and a stop request that is honored
        during startup as well as during the frame loop.
        """

        def __init__(self, *args, model_path_override='', **kwargs):
            super().__init__(*args, **kwargs)
            self.ready_emitted = False
            # base_processor.stop() only clears self.running, and
            # base_processor.run() assigns self.running = True on entry — so a
            # stop requested before/during startup is erased. Latch it here
            # instead; an Event also makes the connect backoff interruptible.
            self._stop_event = threading.Event()
            if model_path_override:
                self.model_path = model_path_override

        @property
        def stop_requested(self):
            return self._stop_event.is_set()

        def stop(self):
            self._stop_event.set()
            super().stop()

        def run(self):
            if self.stop_requested:
                # SIGTERM landed during credential checks / model import.
                self.logger.info("Stop requested before startup; not running")
                return
            super().run()

        def connect_stream(self, max_retries=10, initial_delay=0.5):
            """Connect, honoring a stop requested while we retry.

            The base retry loop runs minutes in the worst case (10 attempts,
            each up to a 10s socket timeout, plus a 1.5x backoff between) and
            never re-checks the stop flag, so a docker stop during connect
            used to wait out the grace and get SIGKILLed — skipping the
            "offline" post and leaving the device ghost-online. Drive the base
            one attempt at a time so a stop lands within one attempt instead.
            """
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                if self.stop_requested:
                    self.logger.info("Stop requested; abandoning stream connect")
                    return False
                if super().connect_stream(max_retries=1, initial_delay=delay):
                    return True
                if attempt < max_retries and self._stop_event.wait(delay):
                    self.logger.info("Stop requested; abandoning stream connect")
                    return False
                delay *= 1.5  # same backoff the base loop applies
            self.logger.error(f"Failed to connect to stream after {max_retries} attempts")
            return False

        def on_start(self):
            try:
                super().on_start()
            except Exception:
                # The base run() loop swallows on_start errors and keeps
                # spinning with a dead model; stop first so the loop exits
                # immediately and the process reports failure (exit 1).
                self.stop()
                raise
            print(READY_MARKER, flush=True)
            self.ready_emitted = True

    return SingleSessionPeopleCounter(
        config.stream_name,
        config.rtsp_url,
        device_info={
            'device_id': config.device_id,
            'password': config.password,
            'service_secret': config.service_secret or None,
            'api_base_url': config.api_base_url,
        },
        frame_queue=None,                 # no orchestrator MJPEG live view
        inference_request_queue=None,     # None => legacy per-process inference
        inference_result_queue=None,
        processor_heartbeat=multiprocessing.Value('d', time.time()),
        inference_slot_name=None,
        model_path_override=config.model_path,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    config = parse_env()

    print(f'[{config.stream_name}] CrowdCount single-session worker starting', flush=True)
    print(f'[{config.stream_name}] API base: {config.api_base_url}', flush=True)
    if config.frame_width or config.frame_height or config.frame_fps:
        print(f'[{config.stream_name}] Upstream-approved format (informational): '
              f'{config.frame_width}x{config.frame_height}@{config.frame_fps}', flush=True)

    # Credential gate: fail fast on bad credentials; warn loudly in probe mode.
    import requests
    from api.crowdcount_api_client import CrowdCountAPIClient

    if config.probe_mode:
        print(f'[{config.stream_name}] '
              'PROBE MODE (no credentials — counts will NOT post)', flush=True)
        logger.warning('Running in PROBE MODE: no CC_SERVICE_SECRET and no '
                       'CC_DEVICE_ID/CC_DEVICE_PASSWORD; crossings are counted '
                       'locally but never posted')
    elif config.service_mode:
        # Trusted-backend service auth: no per-device password to verify here.
        # The online post below is the verification (a bad secret 401s).
        print(f'[{config.stream_name}] SERVICE AUTH for device '
              f'{config.device_id} (counts post via CC_SERVICE_SECRET)', flush=True)
    else:
        try:
            verified = CrowdCountAPIClient.check_password(
                config.device_id, config.password, base_url=config.api_base_url)
        except requests.ConnectionError as e:
            print(f'ERROR: cannot reach {config.api_base_url} to verify credentials '
                  f'for device {config.device_id}: {e}', file=sys.stderr, flush=True)
            sys.exit(3)
        if not verified:
            print(f'ERROR: credentials rejected for device {config.device_id} '
                  f'at {config.api_base_url}', file=sys.stderr, flush=True)
            sys.exit(3)
        print(f'[{config.stream_name}] Credentials verified for device '
              f'{config.device_id}', flush=True)

    # Lifecycle client for online/offline — separate from the processor's own
    # counting client, mirroring the orchestrator's pattern. Disabled (no-op)
    # in probe mode via the client's own credential check.
    lifecycle_client = CrowdCountAPIClient(
        device_id=config.device_id,
        password=config.password,
        base_url=config.api_base_url,
        timeout=LIFECYCLE_API_TIMEOUT_SECONDS,
        service_secret=config.service_secret or None,
    )

    processor = _build_processor(config)

    # Perception overlay (best-effort): if the fanout URL + secret + camera_id
    # are configured, publish live detection boxes so they render on the
    # Upstream dashboard. Purely additive — never affects counting.
    perception = None
    fanout_url = os.environ.get('CC_FANOUT_URL', '').strip()
    fanout_secret = os.environ.get('CC_FANOUT_SECRET', '').strip()
    camera_id = os.environ.get('CC_CAMERA_ID', '').strip()
    if fanout_url and fanout_secret and camera_id:
        try:
            from worker.perception_publisher import PerceptionPublisher
            try:
                fps = float(config.frame_fps)
            except (TypeError, ValueError):
                fps = 15.0
            perception = PerceptionPublisher(
                camera_id=camera_id,
                session_id=os.environ.get('CC_SESSION_ID', '').strip() or None,
                url=fanout_url,
                secret=fanout_secret,
                processor_id=os.environ.get('CC_PROCESSOR_ID', '').strip() or None,
                package_id=os.environ.get('CC_PACKAGE_ID', '').strip() or None,
                fps=fps,
                doorway_center=getattr(processor, 'doorway_config', {}).get('doorway_center'),
            )
            processor.perception_hook = perception
            print(f'[{config.stream_name}] perception overlay ON (camera {camera_id})',
                  flush=True)
        except Exception as e:
            logger.warning(f'perception overlay disabled: {e}')

    def _handle_signal(signum, frame):
        # Cooperative stop only — the run() loop exits within ~a frame read,
        # its finally runs on_stop() (timer cancel + summary) and disconnects.
        # Before the loop, processor.stop_requested makes startup bail out.
        print(f'[{config.stream_name}] Signal {signum} received, stopping...', flush=True)
        processor.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        # max_retries=0: the whole SIGTERM path (this post + the count-thread
        # grace + the offline post) has to fit Upstream's 8s docker stop
        # grace, and retries here are unbounded enough to blow it (~12s) —
        # PEP 475 resumes the retry sleep after the handler runs, so a signal
        # cannot cut it short. Lifecycle posts are best-effort by design.
        try:
            lifecycle_client.update_status('online', max_retries=0)
        except Exception as e:  # best-effort — an API blip must not block counting
            logger.warning(f'online status post failed: {e}')

        # A rejected credential is not a blip: in service mode every count post
        # would 401 while the worker kept reporting itself healthy, silently
        # losing counts-of-record. Fail fast so the supervisor surfaces it.
        # Transient failures leave the verdict None and are tolerated.
        if config.service_mode and lifecycle_client.last_status_auth_ok is False:
            print(f'ERROR: CC_SERVICE_SECRET rejected for device {config.device_id} '
                  f'at {config.api_base_url}; counts would be silently dropped',
                  file=sys.stderr, flush=True)
            sys.exit(3)

        processor.run()  # blocking until stop, stream loss, or init failure
    finally:
        if perception is not None:
            perception.stop()
        # Let in-flight per-crossing count threads finish before exiting.
        time.sleep(COUNT_THREAD_GRACE_SECONDS)
        try:
            lifecycle_client.update_status('offline', max_retries=0)
        except Exception as e:
            logger.warning(f'offline status post failed: {e}')
        sys.stdout.flush()

    if processor.stop_requested or processor.ready_emitted:
        # Requested stop, or ran for real and ended (e.g. stream loss —
        # the supervisor/Docker restart policy owns recovery).
        sys.exit(0)
    print(f'[{config.stream_name}] Worker never reached ready; failing', flush=True)
    sys.exit(1)


if __name__ == '__main__':
    main()
