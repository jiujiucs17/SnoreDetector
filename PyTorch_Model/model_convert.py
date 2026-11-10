import torch
from model_training import SnoreNet
import coremltools as ct
import os

def convert_model(model_path, output_path="", format="mlmodel", model_name="model"):
    if output_path == "":
        output_path = model_path

    trained_model_path = os.path.join(model_path, 'model.pth')

    torch_model = SnoreNet()
    torch_model.load_state_dict(torch.load(trained_model_path))
    torch_model.eval()
    example_input = torch.rand(1, 1, 32, 32)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    if format == "mlmodel":
        ct_model = ct.convert(
            traced_model,
            convert_to="neuralnetwork",
            inputs=[ct.TensorType(shape=example_input.shape)],
        )
        ct_model.save(os.path.join(output_path, model_name + '.mlmodel'))
    elif format == "mlpackage":
        ct_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape)],
        )
        ct_model.save(os.path.join(output_path, model_name + '.mlpackage'))

if __name__ == "__main__":
    model_path = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model"
    output_path = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/coreML_Model"
    convert_model(model_path, output_path=output_path, format="mlmodel", model_name="SnoreNet")