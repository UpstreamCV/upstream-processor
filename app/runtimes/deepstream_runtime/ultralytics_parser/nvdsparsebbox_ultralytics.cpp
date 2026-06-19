/*
 * Upstream Processing P5-K — custom nvinfer bbox parser for Ultralytics
 * (YOLO11 / YOLOv8) detection ONNX exports.
 *
 * ⚠ IN-REPO PREP — NOT YET GPU-VALIDATED. Authored from the published
 * Ultralytics output contract (not copied from any GPL sample). The models
 * that reference this parser are registered ``experimental`` with a
 * "parser authored but not yet GPU-validated" warning; validate detections
 * on a GPU host before promoting any of them to ``probe``.
 *
 * Parses the default (no-NMS) Ultralytics detection export. For COCO-80 the
 * output is a single tensor of shape [1, 84, 8400] (batch stripped by
 * nvinfer -> [84, 8400]): channel-major, where the 84 channels are
 *
 *     cx, cy, w, h, class_0, class_1, ... class_{C-1}
 *
 * Unlike YOLOX there is NO objectness and NO grid/stride decode: the head
 * already emits box centers/sizes in NETWORK-INPUT PIXELS and class scores
 * with sigmoid applied (probabilities in [0, 1]). The score is simply the
 * max class probability. Some exports are anchor-major ([8400, 84]); both
 * orientations are detected from ``num-detected-classes``.
 *
 * Boxes are emitted in network-input pixels (nvinfer rescales to the frame)
 * and thresholded against pre-cluster-threshold; per-class NMS is left to
 * nvinfer's cluster-mode=2 clustering.
 *
 * Build inside the DeepStream 8.0 container (no GPU needed for the compile;
 * see scripts/prepare_ultralytics_model.sh):
 *
 *   g++ -shared -fPIC -O2 -o libnvdsparsebbox_ultralytics.so \
 *       nvdsparsebbox_ultralytics.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -I/usr/local/cuda/include
 */

#include <algorithm>
#include <cstdio>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        fprintf(stderr, "NvDsInferParseUltralytics: no output layers\n");
        return false;
    }
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2) {
        fprintf(stderr, "NvDsInferParseUltralytics: unexpected output rank %d\n",
                layer.inferDims.numDims);
        return false;
    }

    const unsigned int numClasses = detectionParams.numClassesConfigured;
    const unsigned int channels = numClasses + 4;  // cx, cy, w, h + class scores

    /* Two dims (batch stripped). Determine orientation from which dim equals
     * the expected channel count; refuse anything else (a wrong/decoded
     * export would otherwise produce garbage boxes). */
    const int d0 = layer.inferDims.d[0];
    const int d1 = layer.inferDims.d[1];
    long numAnchors = 0;
    bool channelMajor = false;  // true: [channels, anchors]; false: [anchors, channels]
    if (static_cast<unsigned int>(d0) == channels) {
        channelMajor = true;
        numAnchors = d1;
    } else if (static_cast<unsigned int>(d1) == channels) {
        channelMajor = false;
        numAnchors = d0;
    } else {
        fprintf(stderr,
                "NvDsInferParseUltralytics: neither output dim (%d, %d) matches "
                "4+num_classes=%u — is num-detected-classes correct for this "
                "Ultralytics export?\n",
                d0, d1, channels);
        return false;
    }

    const float* data = static_cast<const float*>(layer.buffer);
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (long a = 0; a < numAnchors; ++a) {
        // Accessor for channel c of anchor a, for either memory orientation.
        auto at = [&](unsigned int c) -> float {
            return channelMajor ? data[static_cast<long>(c) * numAnchors + a]
                                : data[a * channels + c];
        };

        unsigned int bestClass = 0;
        float bestProb = at(4);
        for (unsigned int c = 1; c < numClasses; ++c) {
            const float p = at(4 + c);
            if (p > bestProb) {
                bestProb = p;
                bestClass = c;
            }
        }
        const float threshold =
            bestClass < detectionParams.perClassPreclusterThreshold.size()
                ? detectionParams.perClassPreclusterThreshold[bestClass]
                : 0.25f;
        if (bestProb < threshold) {
            continue;
        }

        const float cx = at(0);
        const float cy = at(1);
        const float w = at(2);
        const float h = at(3);
        float left = cx - w / 2.0f;
        float top = cy - h / 2.0f;
        float right = cx + w / 2.0f;
        float bottom = cy + h / 2.0f;
        left = std::max(0.0f, std::min(left, netW - 1.0f));
        top = std::max(0.0f, std::min(top, netH - 1.0f));
        right = std::max(0.0f, std::min(right, netW - 1.0f));
        bottom = std::max(0.0f, std::min(bottom, netH - 1.0f));
        if (right <= left || bottom <= top) {
            continue;
        }

        NvDsInferParseObjectInfo obj;
        obj.classId = bestClass;
        obj.detectionConfidence = bestProb;
        obj.left = left;
        obj.top = top;
        obj.width = right - left;
        obj.height = bottom - top;
        objectList.push_back(obj);
    }
    return true;
}

/* Compile-time prototype check required by nvinfer for custom parsers. */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseUltralytics);
