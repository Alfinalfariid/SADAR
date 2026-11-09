import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class RealtimeDetectionResult {
  final String label;
  final double confidence;
  final double latencyMs;

  // Data diagnostik
  final double rms;
  final String rawLabel;
  final double rawConfidence;
  final String top2Label;
  final double top2Confidence;
  final double confidenceMargin;
  final Map<String, double> probabilities;
  final bool filtered;
  final String decisionReason;

  RealtimeDetectionResult({
    required this.label,
    required this.confidence,
    required this.latencyMs,
    required this.rms,
    required this.rawLabel,
    required this.rawConfidence,
    required this.top2Label,
    required this.top2Confidence,
    required this.confidenceMargin,
    required this.probabilities,
    required this.filtered,
    required this.decisionReason,
  });
}

class RealtimeDetectionService {
  // ============================================
  // AUDIO CONFIGURATION
  // ============================================

  static const int sampleRate = 16000;

  // 0,975 detik
  static const int windowSize = 15600;

  // Window baru setiap sekitar 0,48 detik
  static const int hopSize = 7680;

  static const bool debugDiagnostics = true;

  static const double silenceRmsThreshold = 0.008;

  static const Duration alertHoldDuration = Duration(milliseconds: 2500);

  // ============================================
  // CLASS NAMES
  // ============================================

  static const List<String> classNames = [
    'alarm',
    'background',
    'glass_breaking',
    'scream',
    'siren',
    'vehicle_horn',
  ];

  // ============================================
  // CONFIDENCE THRESHOLDS
  // ============================================

  static const Map<String, double> dangerThresholds = {
    'alarm': 0.70,
    'glass_breaking': 0.80,
    'scream': 0.70,
    'siren': 0.70,

    // V4: dinaikkan dari 0.70 menjadi 0.80.
    // Mengurangi kandidat peringatan klakson palsu.
    'vehicle_horn': 0.80,
  };

  // ============================================
  // CONFIRMATION PER CLASS
  // ============================================

  static const Map<String, int> requiredConfirmations = {
    'alarm': 2,
    'glass_breaking': 1,
    'scream': 2,
    'siren': 2,
    'vehicle_horn': 2,
  };

  // ============================================
  // SERVICES
  // ============================================

  final AudioRecorder _recorder = AudioRecorder();

  Interpreter? _yamnetInterpreter;
  Interpreter? _classifierInterpreter;

  StreamController<RealtimeDetectionResult>? _resultController;

  StreamSubscription<Uint8List>? _audioSubscription;

  // ============================================
  // AUDIO BUFFER
  // ============================================

  final List<double> _audioBuffer = [];

  bool _isRunning = false;

  // ============================================
  // CONFIRMATION STATE
  // ============================================

  String? _candidateLabel;
  int _candidateCount = 0;

  // ============================================
  // LATCH STATE
  // ============================================

  String? _latchedLabel;
  double _latchedConfidence = 0.0;
  DateTime? _latchedUntil;

  // ============================================
  // GETTERS
  // ============================================

  Stream<RealtimeDetectionResult> get results {
    final controller = _resultController;

    if (controller == null) {
      throw StateError('Service belum dimulai.');
    }

    return controller.stream;
  }

  bool get isRunning => _isRunning;

  // ============================================
  // LOAD MODELS
  // ============================================

  Future<void> loadModels() async {
    _yamnetInterpreter = await Interpreter.fromAsset(
      'assets/models/yamnet_embedding_dynamic_int8.tflite',
    );

    _yamnetInterpreter!.resizeInputTensor(0, [windowSize]);

    _yamnetInterpreter!.allocateTensors();

    _classifierInterpreter = await Interpreter.fromAsset(
      'assets/models/danger_classifier_fp32.tflite',
    );

    _classifierInterpreter!.allocateTensors();

    debugPrint('================================');
    debugPrint('REALTIME MODEL INFO');
    debugPrint('================================');

    debugPrint(
      'YAMNet input: '
      '${_yamnetInterpreter!.getInputTensor(0).shape}',
    );

    debugPrint(
      'YAMNet output: '
      '${_yamnetInterpreter!.getOutputTensor(0).shape}',
    );

    debugPrint(
      'Classifier input: '
      '${_classifierInterpreter!.getInputTensor(0).shape}',
    );

    debugPrint(
      'Classifier output: '
      '${_classifierInterpreter!.getOutputTensor(0).shape}',
    );
  }

  // ============================================
  // START DETECTION
  // ============================================

  Future<void> start() async {
    if (_isRunning) {
      return;
    }

    if (_yamnetInterpreter == null || _classifierInterpreter == null) {
      await loadModels();
    }

    final permission = await _recorder.hasPermission();

    if (!permission) {
      throw Exception('Izin microphone tidak diberikan.');
    }

    _audioBuffer.clear();
    _resetCandidate();
    _clearLatch();

    _resultController = StreamController<RealtimeDetectionResult>.broadcast();

    final audioStream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: sampleRate,
        numChannels: 1,
        autoGain: false,
        echoCancel: false,
        noiseSuppress: false,
      ),
    );

    _isRunning = true;

    _audioSubscription = audioStream.listen(
      _handleAudioBytes,
      onError: (error) {
        debugPrint('Audio stream error: $error');
      },
    );
  }

  // ============================================
  // HANDLE AUDIO STREAM
  // ============================================

  void _handleAudioBytes(Uint8List bytes) {
    if (!_isRunning) {
      return;
    }

    final samples = _pcm16ToFloat(bytes);

    _audioBuffer.addAll(samples);

    while (_audioBuffer.length >= windowSize) {
      final window = Float32List.fromList(_audioBuffer.sublist(0, windowSize));

      _processWindow(window);

      if (_audioBuffer.length >= hopSize) {
        _audioBuffer.removeRange(0, hopSize);
      } else {
        _audioBuffer.clear();
      }
    }
  }

  // ============================================
  // PCM16 TO FLOAT32
  // ============================================

  List<double> _pcm16ToFloat(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);

    final sampleCount = bytes.length ~/ 2;

    final samples = List<double>.filled(sampleCount, 0.0);

    for (int i = 0; i < sampleCount; i++) {
      final sample = byteData.getInt16(i * 2, Endian.little);

      samples[i] = sample / 32768.0;
    }

    return samples;
  }

  // ============================================
  // RMS
  // ============================================

  double _calculateRms(Float32List waveform) {
    double sumSquare = 0.0;

    for (final sample in waveform) {
      sumSquare += sample * sample;
    }

    return sqrt(sumSquare / waveform.length);
  }

  // ============================================
  // CANDIDATE
  // ============================================

  void _resetCandidate() {
    _candidateLabel = null;
    _candidateCount = 0;
  }

  void _updateCandidate(String label) {
    if (_candidateLabel == label) {
      _candidateCount++;
    } else {
      _candidateLabel = label;
      _candidateCount = 1;
    }
  }

  // ============================================
  // LATCH
  // ============================================

  bool _isLatchActive() {
    if (_latchedLabel == null || _latchedUntil == null) {
      return false;
    }

    if (DateTime.now().isBefore(_latchedUntil!)) {
      return true;
    }

    _clearLatch();

    return false;
  }

  void _activateLatch({required String label, required double confidence}) {
    _latchedLabel = label;
    _latchedConfidence = confidence;

    _latchedUntil = DateTime.now().add(alertHoldDuration);

    _resetCandidate();
  }

  void _clearLatch() {
    _latchedLabel = null;
    _latchedConfidence = 0.0;
    _latchedUntil = null;
  }

  // ============================================
  // DIAGNOSTIC LOG
  // ============================================

  void _printDiagnostic({
    required double rms,
    required String top1Label,
    required double top1Confidence,
    required String top2Label,
    required double top2Confidence,
    required double margin,
    required Map<String, double> probabilities,
    required String finalLabel,
    required bool filtered,
    required String reason,
  }) {
    if (!debugDiagnostics) {
      return;
    }

    final probabilityText = classNames
        .map((name) => '$name=${(probabilities[name] ?? 0).toStringAsFixed(3)}')
        .join(' | ');

    debugPrint(
      'AUDIO_DEBUG '
      '| RMS=${rms.toStringAsFixed(4)} '
      '| TOP1=$top1Label '
      '${top1Confidence.toStringAsFixed(3)} '
      '| TOP2=$top2Label '
      '${top2Confidence.toStringAsFixed(3)} '
      '| MARGIN=${margin.toStringAsFixed(3)} '
      '| FINAL=$finalLabel '
      '| FILTERED=$filtered '
      '| REASON=$reason',
    );

    debugPrint('PROBS | $probabilityText');
  }

  // ============================================
  // EMIT RESULT
  // ============================================

  void _emitResult({
    required String label,
    required double confidence,
    required double latencyMs,
    required double rms,
    required String rawLabel,
    required double rawConfidence,
    required String top2Label,
    required double top2Confidence,
    required double confidenceMargin,
    required Map<String, double> probabilities,
    required bool filtered,
    required String decisionReason,
  }) {
    _resultController?.add(
      RealtimeDetectionResult(
        label: label,
        confidence: confidence,
        latencyMs: latencyMs,
        rms: rms,
        rawLabel: rawLabel,
        rawConfidence: rawConfidence,
        top2Label: top2Label,
        top2Confidence: top2Confidence,
        confidenceMargin: confidenceMargin,
        probabilities: probabilities,
        filtered: filtered,
        decisionReason: decisionReason,
      ),
    );

    _printDiagnostic(
      rms: rms,
      top1Label: rawLabel,
      top1Confidence: rawConfidence,
      top2Label: top2Label,
      top2Confidence: top2Confidence,
      margin: confidenceMargin,
      probabilities: probabilities,
      finalLabel: label,
      filtered: filtered,
      reason: decisionReason,
    );
  }

  // ============================================
  // PROCESS AUDIO WINDOW
  // ============================================

  void _processWindow(Float32List waveform) {
    final yamnet = _yamnetInterpreter;
    final classifier = _classifierInterpreter;

    if (yamnet == null || classifier == null) {
      return;
    }

    final stopwatch = Stopwatch()..start();

    final rms = _calculateRms(waveform);

    // ==========================================
    // SILENCE FILTER
    // ==========================================

    if (rms < silenceRmsThreshold) {
      stopwatch.stop();

      final emptyProbabilities = <String, double>{
        for (final name in classNames) name: 0.0,
      };

      if (_isLatchActive()) {
        _emitResult(
          label: _latchedLabel!,
          confidence: _latchedConfidence,
          latencyMs: stopwatch.elapsedMicroseconds / 1000.0,
          rms: rms,
          rawLabel: 'silence',
          rawConfidence: 1.0,
          top2Label: '-',
          top2Confidence: 0.0,
          confidenceMargin: 1.0,
          probabilities: emptyProbabilities,
          filtered: true,
          decisionReason: 'silence_but_latch_active',
        );

        return;
      }

      _resetCandidate();

      _emitResult(
        label: 'background',
        confidence: 1.0,
        latencyMs: stopwatch.elapsedMicroseconds / 1000.0,
        rms: rms,
        rawLabel: 'silence',
        rawConfidence: 1.0,
        top2Label: '-',
        top2Confidence: 0.0,
        confidenceMargin: 1.0,
        probabilities: emptyProbabilities,
        filtered: true,
        decisionReason: 'silence_rms',
      );

      return;
    }

    // ==========================================
    // YAMNET EMBEDDING
    // ==========================================

    final embedding = Float32List(1024);

    yamnet.run(waveform, embedding);

    // ==========================================
    // CLASSIFIER
    // ==========================================

    final classifierInput = [embedding.toList()];

    final classifierOutput = [List<double>.filled(classNames.length, 0.0)];

    classifier.run(classifierInput, classifierOutput);

    final probabilitiesList = classifierOutput[0];

    final probabilities = <String, double>{};

    for (int i = 0; i < classNames.length; i++) {
      probabilities[classNames[i]] = probabilitiesList[i];
    }

    // ==========================================
    // TOP-1 AND TOP-2
    // ==========================================

    final rankedIndexes = List<int>.generate(
      classNames.length,
      (index) => index,
    );

    rankedIndexes.sort(
      (a, b) => probabilitiesList[b].compareTo(probabilitiesList[a]),
    );

    final bestIndex = rankedIndexes[0];
    final secondIndex = rankedIndexes[1];

    final rawLabel = classNames[bestIndex];
    final bestScore = probabilitiesList[bestIndex];

    final top2Label = classNames[secondIndex];
    final top2Score = probabilitiesList[secondIndex];

    final margin = bestScore - top2Score;

    stopwatch.stop();

    final latency = stopwatch.elapsedMicroseconds / 1000.0;

    // ==========================================
    // ACTIVE LATCH
    // ==========================================

    if (_isLatchActive()) {
      if (rawLabel == _latchedLabel) {
        final threshold = dangerThresholds[rawLabel] ?? 0.70;

        if (bestScore >= threshold) {
          _latchedUntil = DateTime.now().add(alertHoldDuration);

          if (bestScore > _latchedConfidence) {
            _latchedConfidence = bestScore;
          }
        }
      }

      _emitResult(
        label: _latchedLabel!,
        confidence: _latchedConfidence,
        latencyMs: latency,
        rms: rms,
        rawLabel: rawLabel,
        rawConfidence: bestScore,
        top2Label: top2Label,
        top2Confidence: top2Score,
        confidenceMargin: margin,
        probabilities: probabilities,
        filtered: rawLabel != _latchedLabel,
        decisionReason: 'danger_latch_active',
      );

      return;
    }

    // ==========================================
    // RAW BACKGROUND
    // ==========================================

    if (rawLabel == 'background') {
      _resetCandidate();

      _emitResult(
        label: 'background',
        confidence: bestScore,
        latencyMs: latency,
        rms: rms,
        rawLabel: rawLabel,
        rawConfidence: bestScore,
        top2Label: top2Label,
        top2Confidence: top2Score,
        confidenceMargin: margin,
        probabilities: probabilities,
        filtered: false,
        decisionReason: 'classifier_background',
      );

      return;
    }

    // ==========================================
    // CLASS-SPECIFIC THRESHOLD
    // ==========================================

    final threshold = dangerThresholds[rawLabel] ?? 0.70;

    if (bestScore < threshold) {
      _resetCandidate();

      _emitResult(
        label: 'background',
        confidence: bestScore,
        latencyMs: latency,
        rms: rms,
        rawLabel: rawLabel,
        rawConfidence: bestScore,
        top2Label: top2Label,
        top2Confidence: top2Score,
        confidenceMargin: margin,
        probabilities: probabilities,
        filtered: true,
        decisionReason: 'below_class_threshold',
      );

      return;
    }

    // ==========================================
    // CLASS-SPECIFIC CONFIRMATION
    // ==========================================

    final required = requiredConfirmations[rawLabel] ?? 2;

    _updateCandidate(rawLabel);

    if (_candidateCount < required) {
      _emitResult(
        label: 'background',
        confidence: bestScore,
        latencyMs: latency,
        rms: rms,
        rawLabel: rawLabel,
        rawConfidence: bestScore,
        top2Label: top2Label,
        top2Confidence: top2Score,
        confidenceMargin: margin,
        probabilities: probabilities,
        filtered: true,
        decisionReason: 'waiting_confirmation_${_candidateCount}_of_$required',
      );

      return;
    }

    // ==========================================
    // DANGER CONFIRMED
    // ==========================================

    _activateLatch(label: rawLabel, confidence: bestScore);

    _emitResult(
      label: rawLabel,
      confidence: bestScore,
      latencyMs: latency,
      rms: rms,
      rawLabel: rawLabel,
      rawConfidence: bestScore,
      top2Label: top2Label,
      top2Confidence: top2Score,
      confidenceMargin: margin,
      probabilities: probabilities,
      filtered: false,
      decisionReason: 'danger_confirmed',
    );
  }

  // ============================================
  // STOP DETECTION
  // ============================================

  Future<void> stop() async {
    if (!_isRunning) {
      return;
    }

    _isRunning = false;

    await _audioSubscription?.cancel();
    _audioSubscription = null;

    await _recorder.stop();

    _audioBuffer.clear();

    _resetCandidate();
    _clearLatch();

    await _resultController?.close();
    _resultController = null;
  }

  // ============================================
  // DISPOSE
  // ============================================

  Future<void> dispose() async {
    await stop();

    _yamnetInterpreter?.close();
    _classifierInterpreter?.close();

    _yamnetInterpreter = null;
    _classifierInterpreter = null;

    await _recorder.dispose();
  }
}
