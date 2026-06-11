/*
 * Upstream Processing P5-B — custom nvinfer bbox parser for YOLOX.
 *
 * Parses the RAW (undecoded) output of the official Megvii YOLOX ONNX
 * exports (e.g. yolox_s.onnx from the 0.1.1rc0 release): a single tensor of
 * shape [num_anchors, 5 + num_classes] where each row is
 *
 *     cx, cy (grid-relative), w, h (log-space), objectness, class scores...
 *
 * Decode mirrors YOLOX's demo_postprocess (demo/ONNXRuntime): anchors are
 * laid out stride-major over strides {8, 16, 32} of the network input
 * (640x640 -> 80*80 + 40*40 + 20*20 = 8400 rows):
 *
 *     x = (out[0] + grid_x) * stride        y = (out[1] + grid_y) * stride
 *     w = exp(out[2]) * stride              h = exp(out[3]) * stride
 *     score = objectness * max(class_scores)
 *
 * Boxes are emitted in network-input pixels (nvinfer rescales to the frame)
 * and thresholded against pre-cluster-threshold; per-class NMS is left to
 * nvinfer's cluster-mode=2 clustering. Written from the published output
 * contract (Apache-2.0 YOLOX), not copied from any SDK sample.
 *
 * Build inside the DeepStream 8.0 container (no GPU needed; see
 * scripts/prepare_yolox_model.sh):
 *
 *   g++ -shared -fPIC -O2 -o libnvdsparsebbox_yolox.so \
 *       nvdsparsebbox_yolox.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -I/usr/local/cuda/include
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const int kStrides[] = {8, 16, 32};

extern "C" bool NvDsInferParseYolox(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseYolox(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        fprintf(stderr, "NvDsInferParseYolox: no output layers\n");
        return false;
    }
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2) {
        fprintf(stderr, "NvDsInferParseYolox: unexpected output rank %d\n",
                layer.inferDims.numDims);
        return false;
    }

    /* Last dim = 5 + num_classes; everything before it = anchors. */
    const int lastDim = layer.inferDims.d[layer.inferDims.numDims - 1];
    if (lastDim < 6) {
        fprintf(stderr, "NvDsInferParseYolox: last dim %d too small\n", lastDim);
        return false;
    }
    const unsigned int valuesPerAnchor = static_cast<unsigned int>(lastDim);
    const unsigned int numClasses = valuesPerAnchor - 5;

    long anchorsInTensor = 1;
    for (int i = 0; i < layer.inferDims.numDims - 1; ++i) {
        anchorsInTensor *= layer.inferDims.d[i];
    }
    long anchorsExpected = 0;
    for (int s : kStrides) {
        anchorsExpected += static_cast<long>(networkInfo.width / s) *
                           static_cast<long>(networkInfo.height / s);
    }
    if (anchorsInTensor != anchorsExpected) {
        /* A decoded/dynamic export would break the grid walk below — refuse
         * loudly instead of emitting garbage boxes. */
        fprintf(stderr,
                "NvDsInferParseYolox: anchor mismatch (tensor=%ld expected=%ld "
                "for %ux%u input) — is this a RAW official YOLOX export?\n",
                anchorsInTensor, anchorsExpected, networkInfo.width,
                networkInfo.height);
        return false;
    }

    const float* data = static_cast<const float*>(layer.buffer);
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    long anchor = 0;
    for (int s : kStrides) {
        const int gridW = networkInfo.width / s;
        const int gridH = networkInfo.height / s;
        for (int gy = 0; gy < gridH; ++gy) {
            for (int gx = 0; gx < gridW; ++gx, ++anchor) {
                const float* p = data + anchor * valuesPerAnchor;
                const float objectness = p[4];

                unsigned int bestClass = 0;
                float bestProb = p[5];
                for (unsigned int c = 1; c < numClasses; ++c) {
                    if (p[5 + c] > bestProb) {
                        bestProb = p[5 + c];
                        bestClass = c;
                    }
                }
                const float score = objectness * bestProb;
                const float threshold =
                    bestClass < detectionParams.perClassPreclusterThreshold.size()
                        ? detectionParams.perClassPreclusterThreshold[bestClass]
                        : 0.25f;
                if (score < threshold) {
                    continue;
                }

                const float cx = (p[0] + static_cast<float>(gx)) * s;
                const float cy = (p[1] + static_cast<float>(gy)) * s;
                const float w = std::exp(p[2]) * s;
                const float h = std::exp(p[3]) * s;
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
                obj.detectionConfidence = score;
                obj.left = left;
                obj.top = top;
                obj.width = right - left;
                obj.height = bottom - top;
                objectList.push_back(obj);
            }
        }
    }
    return true;
}

/* Compile-time prototype check required by nvinfer for custom parsers. */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolox);
