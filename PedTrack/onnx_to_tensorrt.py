# onnx_to_tensorrt.py
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#


from __future__ import print_function

import os
import argparse

import tensorrt as trt


EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


def build_engine(onnx_file_path, engine_file_path, verbose=False):
    """Takes an ONNX file and creates a TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        #builder.fp16_mode = True
        #builder.strict_type_constraints = True

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # The actual yolo*.onnx is generated with batch size 64.
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape
        print('Completed parsing of ONNX file')

        print('Building an engine; this may take a while...')
        cfg = builder.create_builder_config()
        cfg.set_flag(trt.BuilderFlag.FP16)
        print(cfg.flags)
        #cfg.max_workspace_size = 1 << 28 # 256MB is for jetson nano
        cfg.max_workspace_size = 1 << 28
        #cfg.flags = trt.BuilderFlag.FP16# | 1 << trt.BuilderFlag.DEBUG)
        engine = builder.build_engine(network,cfg)
        print('Completed creating engine')
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    #parser.add_argument(
    #    '--model', type=str, required=True,
    #    help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
    #          '[{dimension}], where dimension could be a single '
    #          'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument('--onnx_model',type=str,required=True,
        help=('onnx file model_onnx/yolov4-800.onnx'))
    parser.add_argument('--output_engine',type=str,required=True,
        help=('output file path model_tensorRT/yolov4-800.engine'))
    args = parser.parse_args()

    onnx_file_path = args.onnx_model #'%s.onnx' % args.model
    if not os.path.isfile(onnx_file_path):
        raise SystemExit('ERROR: file (%s) not found!  You might want to run yolo_to_onnx.py first to generate it.' % onnx_file_path)
    engine_file_path = args.output_engine  #'%s.engine' % args.model#'%s.trt' % args.model
    _ = build_engine(onnx_file_path, engine_file_path, args.verbose)


if __name__ == '__main__':
    main()
