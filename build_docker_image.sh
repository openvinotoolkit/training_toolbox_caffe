#!/usr/bin/env bash
set -e

display_usage() {
  echo
  echo "Usage: $0 [cpu,gpu] (default: gpu)"
  echo
}

TARGET="${1:-gpu}"
IMAGE_NAME="ttcf"
IMAGE_FULL_NAME="${IMAGE_NAME}:${TARGET}"

OPENVINO_VERSION=$(ls -tF l_openvino_toolkit_p_*.tgz | head -n 1 | grep -oP '(?<=l_openvino_toolkit_p_)\d+\.\d+\.\d+')

case "$TARGET" in
    gpu)
        CUDA_VERSION=$(grep -oP '(?<=CUDA Version )(\d+)' /usr/local/cuda/version.txt)
        if [ -z "$CUDA_VERSION" ];
        then
            echo "[ERROR] Could NOT find CUDA"
            exit 1
        fi
        # https://hub.docker.com/r/nvidia/cuda/
        BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}.0-cudnn7-devel-ubuntu16.04"
        EXEC_BIN="nvidia-docker"
        ;;
    cpu)
        BASE_IMAGE="ubuntu:16.04"
        EXEC_BIN="docker"
        ;;
    *)
        display_usage
        exit 1
        ;;
  esac


if [ -z "$OPENVINO_VERSION" ];
then
    echo "[ERROR] Could NOT find OpenVINO package"
    exit 1
fi

echo ""
echo "Base name:        ${BASE_IMAGE}"
echo "Image name:       ${IMAGE_FULL_NAME}"
echo "OpenVINO version: ${OPENVINO_VERSION}"
echo "Detected CUDA:    ${CUDA_VERSION}"
echo ""

# shellcheck disable=SC2154
$EXEC_BIN build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg OPENVINO_VERSION="${OPENVINO_VERSION}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    -t "${IMAGE_FULL_NAME}" \
    -t "${IMAGE_NAME}:latest" \
    -f docker/Dockerfile \
    .
