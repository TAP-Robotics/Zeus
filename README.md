# Zeus
This codebase handles all server side logic including image inference and command management.

### Training GrabiNet
1. Open the google colab and run each commands chronologically.
2. Download the data.yaml file.
3. Download the best.pt files in the runs folder.
4. Convert the best.pt file to an onnx compatible file, then place it in the models folder.
5. Update the path in the code, to point to the best.onnx and data.yaml files.
