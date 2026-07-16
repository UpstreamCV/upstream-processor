"""
CrowdCount API Client
Handles API calls to the CrowdCount backend server
"""

import hashlib
import hmac as hmac_mod
import json
import logging
import time
import uuid
from typing import Optional

import requests


def _derive_signing_key(device_id: str, password: str) -> str:
    """Derive signing key: hex-encoded SHA256(device_id + ":" + password).

    Must match crowdcount-server's signature_manager.derive_signing_key().
    """
    raw = f"{device_id}:{password}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _derive_service_key(secret: str) -> str:
    """Derive the service signing key: hex-encoded SHA256(secret).

    Must match crowdcount-server's signature_manager._derive_service_key().
    Used for trusted-backend service auth in place of a per-device key.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _compute_signature(
    signing_key: str,
    method: str,
    path: str,
    body: bytes,
    timestamp: str,
    nonce: str,
) -> str:
    """Compute HMAC-SHA256 signature for a request.

    String to sign: METHOD\nPATH\nSHA256(body)\nTIMESTAMP\nNONCE
    Must match crowdcount-server's signature_manager.compute_signature().
    """
    body_hash = hashlib.sha256(body).hexdigest()
    string_to_sign = f"{method}\n{path}\n{body_hash}\n{timestamp}\n{nonce}"
    return hmac_mod.new(
        bytes.fromhex(signing_key),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class CrowdCountAPIClient:
    """
    Client for making API calls to CrowdCount backend
    Each device/stream has its own credentials
    """

    def __init__(self, device_id: str, password: str = '', base_url: str = 'https://api.crowdcount.tech', timeout: int = 5, service_secret: Optional[str] = None):
        """
        Initialize API client.

        Two auth modes:
          * Per-device: device_id + password (the on-device path).
          * Service: a shared service_secret authenticates on behalf of
            device_id without a per-device password (the Upstream worker
            path). Takes precedence when both are supplied.

        Args:
            device_id: Unique device identifier (the target device)
            password: Device password (per-device auth only)
            base_url: Base URL for the CrowdCount API
            timeout: Request timeout in seconds
            service_secret: Shared service credential; enables service auth
        """
        self.device_id = device_id
        self.password = password
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.service_secret = service_secret
        self.use_service_auth = bool(service_secret)
        # Enabled if we can authenticate: service secret, or device+password.
        self.enabled = self.use_service_auth or bool(device_id and password)

        # Derive the signing key once at init (service key or per-device key).
        if self.use_service_auth:
            self._signing_key = _derive_service_key(service_secret)
        elif self.enabled:
            self._signing_key = _derive_signing_key(device_id, password)
        else:
            self._signing_key = ""

        # Setup logging (only add handler and log once per device_id)
        self.logger = logging.getLogger(f"APIClient-{device_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - {device_id} - API - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            # Only log initialization once per device
            if self.enabled:
                self.logger.info(f"API Client initialized for device: {device_id}")
            else:
                self.logger.info(f"API Client disabled (no credentials provided)")

    def _signed_request(self, path: str, data: dict, idempotent: bool = False) -> requests.Response:
        """Send a signed POST request with optional idempotency key.

        Serializes the body, computes HMAC-SHA256 signature, and attaches
        all required headers for phase-3 signature verification.

        Args:
            path: API path (e.g. "/add_to_count")
            data: JSON payload dict
            idempotent: Whether to attach an Idempotency-Key header

        Returns:
            The requests.Response object
        """
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        ts = str(int(time.time()))
        nonce = uuid.uuid4().hex

        sig = _compute_signature(self._signing_key, "POST", path, body, ts, nonce)

        sig_header = "X-CC-Service-Signature" if self.use_service_auth else "X-CC-Signature"
        headers = {
            "Content-Type": "application/json",
            sig_header: sig,
            "X-CC-Timestamp": ts,
            "X-CC-Nonce": nonce,
            "X-CC-Device-Id": self.device_id,
        }
        if idempotent:
            headers["Idempotency-Key"] = uuid.uuid4().hex

        return requests.post(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            timeout=self.timeout,
        )

    def add_to_count(self, count: int = 1, max_retries: int = 2) -> bool:
        """
        Add to the count (person entered)

        Args:
            count: Number to add to count (default: 1)
            max_retries: Maximum number of retries on transient failure (default: 2)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return True  # Silently succeed if API is disabled

        data = {
            'device_id': self.device_id,
            'password': self.password,
            'count_to_add': str(count)
        }

        for attempt in range(1 + max_retries):
            try:
                self.logger.debug(f"POST /add_to_count - Adding {count} to count (attempt {attempt + 1})")
                response = self._signed_request("/add_to_count", data, idempotent=True)

                if response.status_code == 200:
                    self.logger.info(f"✓ Added {count} to remote count")
                    return True
                elif response.status_code == 503:
                    raise requests.ConnectionError("Server starting up (503)")
                else:
                    self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                    return False  # Non-transient error, don't retry

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)  # 1s, 2s
                    self.logger.warning(f"✗ API {type(e).__name__} (attempt {attempt + 1}/{1 + max_retries}), retrying in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"✗ API {type(e).__name__}: Failed after {1 + max_retries} attempts")
                    return False
            except Exception as e:
                self.logger.warning(f"✗ API Error: {e}")
                return False

        return False

    def subtract_from_count(self, count: int = 1, max_retries: int = 2) -> bool:
        """
        Subtract from the count (person exited)

        Args:
            count: Number to subtract from count (default: 1)
            max_retries: Maximum number of retries on transient failure (default: 2)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return True  # Silently succeed if API is disabled

        data = {
            'device_id': self.device_id,
            'password': self.password,
            'count_to_sub': str(count)
        }

        for attempt in range(1 + max_retries):
            try:
                self.logger.debug(f"POST /sub_from_count - Subtracting {count} from count (attempt {attempt + 1})")
                response = self._signed_request("/sub_from_count", data, idempotent=True)

                if response.status_code == 200:
                    self.logger.info(f"✓ Subtracted {count} from remote count")
                    return True
                elif response.status_code == 503:
                    raise requests.ConnectionError("Server starting up (503)")
                else:
                    self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                    return False  # Non-transient error, don't retry

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)  # 1s, 2s
                    self.logger.warning(f"✗ API {type(e).__name__} (attempt {attempt + 1}/{1 + max_retries}), retrying in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"✗ API {type(e).__name__}: Failed after {1 + max_retries} attempts")
                    return False
            except Exception as e:
                self.logger.warning(f"✗ API Error: {e}")
                return False

        return False

    def get_current_count(self) -> Optional[int]:
        """
        Get the current count from the API

        Returns:
            Current count as integer, or None if request failed
        """
        if not self.enabled:
            return None

        try:
            data = {
                'device_id': self.device_id,
                'password': self.password
            }

            response = self._signed_request("/get_count", data)

            if response.status_code == 200:
                resp_data = response.json()
                count = resp_data.get('count')
                self.logger.debug(f"Current count: {count}")
                return count
            else:
                self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                return None

        except Exception as e:
            self.logger.warning(f"✗ API Error: {e}")
            return None

    def set_enabled(self, enabled: bool):
        """
        Enable or disable API calls

        Args:
            enabled: True to enable, False to disable
        """
        self.enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"API calls {status}")

    def update_status(self, status: str, max_retries: int = 2) -> bool:
        """
        Update device online/offline status on the server

        Args:
            status: Either "online" or "offline"
            max_retries: Maximum number of retries on transient failure (default: 2)

        Returns:
            True if successful, False otherwise
        """
        if status not in {"online", "offline"}:
            self.logger.error(f"Invalid status: {status}. Must be 'online' or 'offline'")
            return False

        # Always try to update status, even if API is "disabled" for counting
        # Status updates are important for device management
        if not self.device_id or not (self.password or self.use_service_auth):
            self.logger.warning("Cannot update status: no credentials")
            return False

        data = {
            'device_id': self.device_id,
            'password': self.password,
            'status': status
        }

        for attempt in range(1 + max_retries):
            try:
                self.logger.debug(f"POST /update_status - Updating status to {status} (attempt {attempt + 1})")
                response = self._signed_request("/update_status", data, idempotent=True)

                if response.status_code == 200:
                    self.logger.info(f"✓ Device status updated to {status}")
                    return True
                elif response.status_code == 503:
                    raise requests.ConnectionError("Server starting up (503)")
                else:
                    self.logger.warning(f"✗ Status Update Error: HTTP {response.status_code}")
                    return False

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)  # 1s, 2s
                    self.logger.warning(f"✗ Status Update {type(e).__name__} (attempt {attempt + 1}/{1 + max_retries}), retrying in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"✗ Status Update {type(e).__name__}: Failed after {1 + max_retries} attempts")
                    return False
            except Exception as e:
                self.logger.warning(f"✗ Status Update Error: {e}")
                return False

        return False

    def get_direction_flipped(self, max_retries: int = 2) -> bool:
        """
        Get the direction_flipped setting from the server

        Args:
            max_retries: Maximum number of retries on transient failure (default: 2)

        Returns:
            True if direction is flipped, False otherwise (defaults to False on error)
        """
        if not self.enabled:
            return False

        data = {
            'device_id': self.device_id,
            'password': self.password
        }

        for attempt in range(1 + max_retries):
            try:
                response = self._signed_request("/get_direction_flipped", data)

                if response.status_code == 200:
                    resp_data = response.json()
                    flipped = resp_data.get('direction_flipped', False)
                    self.logger.debug(f"Direction flipped setting: {flipped}")
                    return bool(flipped)
                elif response.status_code == 503:
                    raise requests.ConnectionError("Server starting up (503)")
                else:
                    self.logger.warning(f"✗ Failed to get direction_flipped: HTTP {response.status_code}")
                    return False

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)  # 1s, 2s
                    self.logger.warning(f"✗ get_direction_flipped {type(e).__name__} (attempt {attempt + 1}/{1 + max_retries}), retrying in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"✗ get_direction_flipped {type(e).__name__}: Failed after {1 + max_retries} attempts")
                    return False
            except Exception as e:
                self.logger.warning(f"✗ Error getting direction_flipped: {e}")
                return False

        return False

    @staticmethod
    def check_password(device_id: str, password: str, base_url: str = 'https://api.crowdcount.tech', timeout: int = 10) -> bool:
        """
        Check if a password is correct for a given device_id.
        Retries on timeout/connection errors (server may be starting up).

        This is a static method used before a client instance exists,
        so it signs the request inline.

        Args:
            device_id: Device identifier
            password: Password to validate
            base_url: Base URL for the CrowdCount API
            timeout: Request timeout in seconds

        Returns:
            True if password is correct, False otherwise
        """
        base_url = base_url.rstrip('/')
        path = "/check_password"
        data = {
            'device_id': device_id,
            'password': password
        }
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                # Fresh timestamp/nonce each attempt
                ts = str(int(time.time()))
                nonce = uuid.uuid4().hex
                signing_key = _derive_signing_key(device_id, password)
                sig = _compute_signature(signing_key, "POST", path, body, ts, nonce)

                headers = {
                    "Content-Type": "application/json",
                    "X-CC-Signature": sig,
                    "X-CC-Timestamp": ts,
                    "X-CC-Nonce": nonce,
                    "X-CC-Device-Id": device_id,
                }

                response = requests.post(
                    f"{base_url}{path}",
                    data=body,
                    headers=headers,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    resp_data = response.json()
                    return bool(resp_data.get('is_password_correct', False))
                elif response.status_code == 503:
                    # Server starting up — retry
                    raise requests.ConnectionError("Server starting up (503)")
                elif response.status_code == 404:
                    return False
                else:
                    return False

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_attempts:
                    print(f"✗ check_password timeout (attempt {attempt}/{max_attempts}), retrying in {attempt}s", flush=True)
                    time.sleep(attempt)
                    continue
                print(f"✗ check_password: Failed after {max_attempts} attempts", flush=True)
                raise requests.ConnectionError(f"API unreachable after {max_attempts} attempts") from e
            except Exception:
                return False
