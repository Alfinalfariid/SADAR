import 'dart:math';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class MemoryBenchmarkService {
  static const int windowSize = 15600;

  Interpreter? _interpreter;

  String? loadedModel;

  Future<void> loadModel({
    required String modelPath,
    required String modelName,
  }) async {
    closeModel();

    final interpreter = await Interpreter.fromAsset(
      modelPath,
    );

    interpreter.resizeInputTensor(
      0,
      [windowSize],
    );

    interpreter.allocateTensors();

    final input = Float32List(windowSize);

    for (int i = 0; i < windowSize; i++) {
      input[i] = sin(
        2 * pi * 440 * i / 16000,
      );
    }

    final output = Float32List(1024);

    // Warmup supaya kondisi model stabil
    for (int i = 0; i < 20; i++) {
      interpreter.run(
        input,
        output,
      );
    }

    _interpreter = interpreter;
    loadedModel = modelName;
  }

  void closeModel() {
    _interpreter?.close();
    _interpreter = null;
    loadedModel = null;
  }

  void dispose() {
    closeModel();
  }
}