# MiniFASNetV2 model

- Model: MiniFASNetV2, ONNX
- Architecture and original weights: <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
- ONNX conversion: <https://github.com/yakhyo/face-anti-spoofing>
- Release asset: `MiniFASNetV2.onnx`
- License: Apache License 2.0
- SHA-256: `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`

The model expects an unnormalized BGR tensor shaped `[1, 3, 80, 80]`. Class 1
is treated as the bona-fide face class. The face crop uses a 2.7× scale.
