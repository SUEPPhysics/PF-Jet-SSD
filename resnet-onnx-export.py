import argparse
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import yaml

from numpy.testing import assert_almost_equal as is_equal
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static)
from resnet.net import build_resnet
from utils import *


class DataReader(CalibrationDataReader):
    def __init__(self, batch_size, data_loader, model_path):
        self.datasize = 10
        self.data_loader = data_loader
        self.model_path = model_path
        self.enum_data = []
        self.data = np.zeros(
            (self.datasize, batch_size, 1, 280, 360),
            dtype=np.float32)
        self.batch_size = batch_size
        self.foo()

    def foo(self):
        for i in range(self.datasize):
            images = []
            for batch_index, (image, _) in enumerate(self.data_loader):
                if batch_index < self.batch_size:
                    images.append(to_numpy(image))
                else:
                    break
            self.data[i] = np.ascontiguousarray(images, dtype=np.float32)
        session = onnxruntime.InferenceSession(self.model_path, None)
        input_name = session.get_inputs()[0].name
        self.enum_data = iter([{input_name: d} for d in self.data])

    def get_next(self):
        return next(self.enum_data, None)


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert PyTorch ResNet to ONNX")
    parser.add_argument('model',
                        type=str,
                        help='Input model name')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        help='Test batch size',
                        default=1)
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='ssd-config.yml')
    parser.add_argument('-s', '--suppress',
                        action='store_true',
                        help='Suppress checks')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    
    logger = set_logging('Test_ResNet',
                         '{}/PF-Jet-ResNet-Test.log'.format(
                              config['output']['model']),
                         args.verbose)

    logger.info('Converting {} model to ONNX'.format(args.model))

    
    dataset = config['dataset']['validation'][0]
    input_dimensions = config['resnet_settings']['input_dimensions']

    base = '{}/{}'.format(config['output']['model'], args.model)
    source_path = '{}.pth'.format(base)
    export_path = '{}.onnx'.format(base)
    export_int8_path = '{}-int8.onnx'.format(base)

    torch.set_default_tensor_type('torch.FloatTensor')

    logger.info('Prepare PyTorch model')
    net = build_resnet(input_dimensions=input_dimensions,
                   file_path='models/ResNet18-ABCDSingle.pth',
                   rank=torch.device('cpu'))
    net.eval()

    logger.info('Prepare inputs')
    loader = get_data_loader(dataset,
                             args.batch_size,
                             config['evaluation_pref']['workers'],
                             input_dimensions,
                             1,
                             cpu=True,
                             shuffle=False)

    batch_iterator = iter(loader)
    dummy_input, _ = next(batch_iterator)
    logger.info('Export as ONNX model')
    torch.onnx.export(net,
                      dummy_input,
                      export_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    logger.info('Validating graph')
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)

    logger.info('Matching outputs')
    ort_session = onnxruntime.InferenceSession(export_path)
    # Compute PyTorch output prediction
    torch_out = list(map(to_numpy, list(net(dummy_input))))
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_out = ort_session.run(None, ort_inputs)
    # Compare ONNX Runtime and PyTorch results
    if not args.suppress:
        logger.info('Performing checks')
        for torch_event, ort_event in zip(torch_out, ort_out):
            ort_event = ort_event.squeeze(0)
            is_equal(torch_event, ort_event, decimal=3)
        logger.info("All good")

    logger.info("Exported model has been successfully tested with ONNXRuntime")
