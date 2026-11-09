import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class YamNetPrediction {
  final String label;
  final double score;

  YamNetPrediction({
    required this.label,
    required this.score,
  });
}

class YamNetService {
  static const int frameSize = 15600;

  Interpreter? _interpreter;

  List<String> _labels = [];

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    // ============================================
    // OPTIONS
    // ============================================

    final options = InterpreterOptions();

    options.threads = 2;

    // ============================================
    // LOAD MODEL
    // ============================================

    _interpreter = await Interpreter.fromAsset(
      'assets/models/yamnet.tflite',
      options: options,
    );

    // ============================================
    // CEK INPUT AWAL
    // ============================================

    final originalInput =
        _interpreter!.getInputTensor(0);

    print(
      'YAMNet input awal: ${originalInput.shape}',
    );

    // Model kita sebelumnya di Python:
    //
    // shape awal      : [1]
    // shape signature : [-1]
    //
    // Resize menjadi 15600 sample.
    _interpreter!.resizeInputTensor(
      0,
      [frameSize],
    );

    // Wajib allocate ulang sesudah resize.
    _interpreter!.allocateTensors();

    // ============================================
    // CEK INPUT / OUTPUT SETELAH RESIZE
    // ============================================

    final inputTensor =
        _interpreter!.getInputTensor(0);

    final outputTensor =
        _interpreter!.getOutputTensor(0);

    print(
      'YAMNet input setelah resize: '
      '${inputTensor.shape}',
    );

    print(
      'YAMNet output: '
      '${outputTensor.shape}',
    );

    // ============================================
    // LOAD LABEL
    // ============================================

    final labelText = await rootBundle.loadString(
      'assets/labels/yamnet_521_labels.txt',
    );

    _labels = labelText
        .split('\n')
        .map(
          (label) => label.trim(),
        )
        .where(
          (label) => label.isNotEmpty,
        )
        .toList();

    print(
      'Jumlah label YAMNet: ${_labels.length}',
    );

    if (_labels.length != 521) {
      throw Exception(
        'Jumlah label YAMNet bukan 521.',
      );
    }
  }

  // ============================================
  // TEST SILENCE
  // ============================================

  List<YamNetPrediction> testSilence() {
    if (_interpreter == null) {
      throw Exception(
        'YAMNet belum dimuat.',
      );
    }

    // 15600 sample bernilai 0
    // berarti silence.
    final input = List<double>.filled(
      frameSize,
      0.0,
    );

    // Output YAMNet:
    // [1, 521]
    final output = [
      List<double>.filled(
        521,
        0.0,
      ),
    ];

    final stopwatch = Stopwatch()
      ..start();

    _interpreter!.run(
      input,
      output,
    );

    stopwatch.stop();

    print(
      'Inference latency: '
      '${stopwatch.elapsedMicroseconds / 1000} ms',
    );

    final scores = output[0];

    final indexedScores =
        List.generate(
      scores.length,
      (index) => MapEntry(
        index,
        scores[index],
      ),
    );

    indexedScores.sort(
      (a, b) =>
          b.value.compareTo(a.value),
    );

    return indexedScores
        .take(5)
        .map(
          (entry) => YamNetPrediction(
            label: _labels[entry.key],
            score: entry.value,
          ),
        )
        .toList();
  }

  void close() {
    _interpreter?.close();

    _interpreter = null;
  }
}