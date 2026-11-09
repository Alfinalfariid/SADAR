import 'dart:math';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class BenchmarkResult {
  final String modelName;
  final double meanMs;
  final double stdMs;
  final double minMs;
  final double maxMs;
  final double medianMs;

  BenchmarkResult({
    required this.modelName,
    required this.meanMs,
    required this.stdMs,
    required this.minMs,
    required this.maxMs,
    required this.medianMs,
  });
}

class ModelBenchmarkService {
  static const int windowSize = 15600;

  Future<BenchmarkResult> benchmarkYamNet({
    required String modelPath,
    required String modelName,
    int warmupRuns = 20,
    int measuredRuns = 100,
  }) async {
    final interpreter = await Interpreter.fromAsset(
      modelPath,
    );

    // Model memiliki dynamic input [-1],
    // tetapi benchmark memakai window tetap 15600.
    interpreter.resizeInputTensor(
      0,
      [windowSize],
    );

    interpreter.allocateTensors();

    print('========================================');
    print(modelName);
    print('========================================');

    print(
      'Input shape : '
      '${interpreter.getInputTensor(0).shape}',
    );

    print(
      'Input type  : '
      '${interpreter.getInputTensor(0).type}',
    );

    print(
      'Output shape: '
      '${interpreter.getOutputTensor(0).shape}',
    );

    print(
      'Output type : '
      '${interpreter.getOutputTensor(0).type}',
    );

    /*
      Input benchmark.

      Nilainya dibuat seperti waveform audio
      dalam rentang -1 sampai 1.

      Data yang sama digunakan FP32 dan INT8,
      sehingga perbandingan tetap adil.
    */
    final input = Float32List(windowSize);

    for (int i = 0; i < windowSize; i++) {
      input[i] = sin(
        2 * pi * 440 * i / 16000,
      ).toDouble();
    }

    final output = Float32List(1024);

    // WARMUP
    for (int i = 0; i < warmupRuns; i++) {
      interpreter.run(
        input,
        output,
      );
    }

    print(
      'Warmup selesai: $warmupRuns kali',
    );

    final List<double> latencies = [];

    // BENCHMARK
    for (int i = 0; i < measuredRuns; i++) {
      final stopwatch = Stopwatch()
        ..start();

      interpreter.run(
        input,
        output,
      );

      stopwatch.stop();

      final elapsedMs =
          stopwatch.elapsedMicroseconds /
              1000.0;

      latencies.add(
        elapsedMs,
      );
    }

    interpreter.close();

    latencies.sort();

    final mean =
        latencies.reduce(
              (a, b) => a + b,
            ) /
            latencies.length;

    final variance =
        latencies
            .map(
              (x) => pow(
                x - mean,
                2,
              ).toDouble(),
            )
            .reduce(
              (a, b) => a + b,
            ) /
        latencies.length;

    final std =
        sqrt(
          variance,
        );

    final minLatency =
        latencies.first;

    final maxLatency =
        latencies.last;

    double median;

    if (latencies.length.isEven) {
      final middle =
          latencies.length ~/ 2;

      median =
          (
                latencies[middle - 1] +
                latencies[middle]
              ) /
              2;
    } else {
      median =
          latencies[
            latencies.length ~/ 2
          ];
    }

    return BenchmarkResult(
      modelName: modelName,
      meanMs: mean,
      stdMs: std,
      minMs: minLatency,
      maxMs: maxLatency,
      medianMs: median,
    );
  }

  Future<List<BenchmarkResult>>
  benchmarkBothModels() async {
    final fp32 =
        await benchmarkYamNet(
          modelPath:
              'assets/models/yamnet_embedding_fp32.tflite',
          modelName:
              'YAMNet FP32',
        );

    final dynamicInt8 =
        await benchmarkYamNet(
          modelPath:
              'assets/models/yamnet_embedding_dynamic_int8.tflite',
          modelName:
              'YAMNet Dynamic INT8',
        );

    return [
      fp32,
      dynamicInt8,
    ];
  }
}