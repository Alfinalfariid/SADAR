import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ThroughputApp());
}

class ThroughputApp extends StatelessWidget {
  const ThroughputApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SADAR Throughput Benchmark',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const ThroughputPage(),
    );
  }
}

class ThroughputPage extends StatefulWidget {
  const ThroughputPage({super.key});

  @override
  State<ThroughputPage> createState() => _ThroughputPageState();
}

class _ThroughputPageState extends State<ThroughputPage> {
  static const int windowSize = 15600;
  static const int sampleRate = 16000;
  static const int warmupRuns = 20;
  static const int measuredRuns = 100;
  static const int repetitions = 3;

  static const Map<String, String> models = {
    'FP32': 'assets/models/yamnet_embedding_fp32.tflite',
    'Dynamic_INT8': 'assets/models/yamnet_embedding_dynamic_int8.tflite',
  };

  bool isRunning = false;
  bool isComplete = false;

  String status = 'Benchmark belum dimulai.';
  String summary = '';
  String csvContent = '';
  String csvPath = '';
  String jsonPath = '';

  final List<Map<String, dynamic>> results = [];

  // ============================================
  // INPUT AUDIO SINTETIS
  // ============================================

  Float32List createInput() {
    final samples = Float32List(windowSize);

    for (int i = 0; i < windowSize; i++) {
      samples[i] = sin(2 * pi * 440 * i / sampleRate);
    }

    return samples;
  }

  // ============================================
  // UKUR THROUGHPUT SATU RUN
  // ============================================

  Future<Map<String, dynamic>> measureModel({
    required String modelName,
    required String modelPath,
    required int run,
    required int sequence,
  }) async {
    final interpreter = await Interpreter.fromAsset(modelPath);

    try {
      interpreter.resizeInputTensor(0, [windowSize]);

      interpreter.allocateTensors();

      final inputInfo = interpreter.getInputTensor(0);

      final outputInfo = interpreter.getOutputTensor(0);

      if (inputInfo.type != TensorType.float32 ||
          outputInfo.type != TensorType.float32) {
        throw StateError('Input/output bukan Float32: $modelName');
      }

      if (inputInfo.shape.length != 1 ||
          inputInfo.shape[0] != windowSize ||
          outputInfo.shape.length != 1 ||
          outputInfo.shape[0] != 1024) {
        throw StateError('Shape tensor tidak sesuai: $modelName');
      }

      final input = createInput();
      final output = Float32List(1024);

      // Warmup tidak ikut dihitung.
      for (int i = 0; i < warmupRuns; i++) {
        interpreter.run(input, output);
      }

      // Mengukur waktu TOTAL untuk
      // menyelesaikan 100 inferensi.

      final stopwatch = Stopwatch()..start();

      for (int i = 0; i < measuredRuns; i++) {
        interpreter.run(input, output);
      }

      stopwatch.stop();

      final totalMs = stopwatch.elapsedMicroseconds / 1000.0;

      if (totalMs <= 0) {
        throw StateError('Waktu pengukuran tidak valid.');
      }

      final totalSeconds = totalMs / 1000.0;

      final throughputIps = measuredRuns / totalSeconds;

      final meanMs = totalMs / measuredRuns;

      return {
        'run': run,
        'sequence': sequence,
        'model': modelName,
        'warmup_runs': warmupRuns,
        'measured_runs': measuredRuns,
        'total_elapsed_ms': totalMs,
        'mean_inference_ms': meanMs,
        'throughput_ips': throughputIps,
      };
    } finally {
      interpreter.close();
    }
  }

  // ============================================
  // THROUGHPUT AGREGAT
  // ============================================

  double aggregateIps(String modelName) {
    final selected = results.where((item) => item['model'] == modelName);

    int totalInferences = 0;
    double totalElapsedMs = 0.0;

    for (final item in selected) {
      totalInferences += item['measured_runs'] as int;

      totalElapsedMs += item['total_elapsed_ms'] as double;
    }

    if (totalElapsedMs <= 0) {
      return 0.0;
    }

    return totalInferences / (totalElapsedMs / 1000.0);
  }

  // ============================================
  // BUAT CSV
  // ============================================

  String buildCsv() {
    final lines = <String>[
      'run,sequence,model,warmup_runs,'
          'measured_runs,total_elapsed_ms,'
          'mean_inference_ms,throughput_ips',
    ];

    for (final row in results) {
      lines.add(
        [
          row['run'],
          row['sequence'],
          row['model'],
          row['warmup_runs'],
          row['measured_runs'],
          (row['total_elapsed_ms'] as double).toStringAsFixed(6),
          (row['mean_inference_ms'] as double).toStringAsFixed(6),
          (row['throughput_ips'] as double).toStringAsFixed(6),
        ].join(','),
      );
    }

    return '${lines.join('\n')}\n';
  }

  // ============================================
  // SIMPAN CSV DAN JSON
  // ============================================

  Future<void> saveResults({
    required double fp32Ips,
    required double int8Ips,
  }) async {
    final timestamp = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');

    final directory = Directory.systemTemp;

    final csvFile = File(
      '${directory.path}/'
      'sadar_throughput_$timestamp.csv',
    );

    final jsonFile = File(
      '${directory.path}/'
      'sadar_throughput_$timestamp.json',
    );

    csvContent = buildCsv();

    await csvFile.writeAsString(csvContent, flush: true);

    final metadata = {
      'experiment': 'SADAR YAMNet Throughput',
      'created_at': DateTime.now().toIso8601String(),
      'platform': Platform.operatingSystem,
      'platform_version': Platform.operatingSystemVersion,
      'metric': 'Inferences per Second',
      'metric_abbreviation': 'IPS',
      'not_server_rps': true,
      'input_type': 'Synthetic sine waveform 440 Hz',
      'sample_rate': sampleRate,
      'window_samples': windowSize,
      'warmup_runs_per_model': warmupRuns,
      'measured_runs_per_model': measuredRuns,
      'repetitions_per_model': repetitions,
      'measurement_scope':
          'Total wall-clock time around '
          '100 sequential Interpreter.run() calls; '
          'excludes model loading, warmup, '
          'audio capture, classifier, and UI',
      'model_order': [
        ['FP32', 'Dynamic_INT8'],
        ['Dynamic_INT8', 'FP32'],
        ['FP32', 'Dynamic_INT8'],
      ],
      'fp32_aggregate_ips': fp32Ips,
      'dynamic_int8_aggregate_ips': int8Ips,
      'results': results,
    };

    await jsonFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(metadata),
      flush: true,
    );

    csvPath = csvFile.path;
    jsonPath = jsonFile.path;

    debugPrint('SADAR_THROUGHPUT_CSV: $csvPath');

    debugPrint('SADAR_THROUGHPUT_JSON: $jsonPath');
  }

  // ============================================
  // MAIN BENCHMARK
  // ============================================

  Future<void> startBenchmark() async {
    if (isRunning) {
      return;
    }

    setState(() {
      isRunning = true;
      isComplete = false;
      results.clear();

      csvContent = '';
      csvPath = '';
      jsonPath = '';
      summary = '';

      status = 'Menyiapkan benchmark...';
    });

    try {
      int sequence = 0;

      for (int run = 1; run <= repetitions; run++) {
        // Urutan run kedua dibalik
        // untuk mengurangi efek urutan model.

        final order = run.isEven
            ? ['Dynamic_INT8', 'FP32']
            : ['FP32', 'Dynamic_INT8'];

        for (final modelName in order) {
          sequence++;

          if (!mounted) {
            return;
          }

          setState(() {
            status =
                'Run $run/$repetitions\n'
                'Model: $modelName\n'
                'Pengukuran $sequence/6';
          });

          await Future<void>.delayed(const Duration(milliseconds: 300));

          final result = await measureModel(
            modelName: modelName,
            modelPath: models[modelName]!,
            run: run,
            sequence: sequence,
          );

          results.add(result);

          debugPrint(
            'THROUGHPUT_RESULT '
            'run=$run '
            'model=$modelName '
            'IPS=${result['throughput_ips']}',
          );

          if (mounted) {
            setState(() {
              status =
                  '$modelName run $run selesai\n'
                  'Throughput: '
                  '${(result['throughput_ips'] as double).toStringAsFixed(2)} IPS';
            });
          }

          await Future<void>.delayed(const Duration(milliseconds: 500));
        }
      }

      final fp32Ips = aggregateIps('FP32');
      final int8Ips = aggregateIps('Dynamic_INT8');

      final improvement = ((int8Ips - fp32Ips) / fp32Ips) * 100.0;

      summary =
          'YAMNet FP32\n'
          '${fp32Ips.toStringAsFixed(2)} IPS\n\n'
          'YAMNet Dynamic INT8\n'
          '${int8Ips.toStringAsFixed(2)} IPS\n\n'
          'Peningkatan throughput\n'
          '${improvement.toStringAsFixed(2)}%\n\n'
          '3 run per model, '
          '100 inferensi terukur per run.';

      await saveResults(fp32Ips: fp32Ips, int8Ips: int8Ips);

      if (mounted) {
        setState(() {
          isComplete = true;
          status = 'Benchmark throughput selesai.';
        });
      }
    } catch (error, stackTrace) {
      debugPrint('Throughput error: $error');
      debugPrintStack(stackTrace: stackTrace);

      if (mounted) {
        setState(() {
          status = 'Benchmark gagal:\n$error';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          isRunning = false;
        });
      }
    }
  }

  // ============================================
  // SALIN CSV
  // ============================================

  Future<void> copyCsv() async {
    if (csvContent.isEmpty) {
      return;
    }

    await Clipboard.setData(ClipboardData(text: csvContent));

    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context)
        .showSnackBar(const SnackBar(content: Text('CSV throughput disalin.')));
  }

  // ============================================
  // USER INTERFACE
  // ============================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SADAR Throughput'), centerTitle: true),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Icon(Icons.speed_rounded, size: 64, color: Colors.indigo),

            const SizedBox(height: 12),

            const Text(
              'YAMNet FP32 vs Dynamic INT8',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 21, fontWeight: FontWeight.bold),
            ),

            const SizedBox(height: 8),

            const Text(
              'Pengukuran throughput inferensi '
              'langsung pada perangkat Android.',
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 16),

            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Konfigurasi pengujian',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    Text('3 run per model'),
                    Text('20 warmup per run'),
                    Text('100 inferensi per run'),
                    Text('Input sinus 440 Hz'),
                    Text('Output: IPS'),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            SizedBox(
              height: 54,
              child: FilledButton.icon(
                onPressed: isRunning ? null : startBenchmark,
                icon: Icon(
                  isRunning
                      ? Icons.hourglass_top_rounded
                      : Icons.play_arrow_rounded,
                ),
                label: Text(isRunning ? 'SEDANG MENGUJI' : 'MULAI BENCHMARK'),
              ),
            ),

            const SizedBox(height: 16),

            if (isRunning) const Center(child: CircularProgressIndicator()),

            Text(status, textAlign: TextAlign.center),

            if (isComplete) ...[
              const SizedBox(height: 20),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: SelectableText(
                    summary,
                    style: const TextStyle(fontSize: 16),
                  ),
                ),
              ),

              const SizedBox(height: 12),

              OutlinedButton.icon(
                onPressed: copyCsv,
                icon: const Icon(Icons.copy_rounded),
                label: const Text('SALIN HASIL CSV'),
              ),

              const SizedBox(height: 12),

              const Text(
                'Lokasi CSV:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),

              SelectableText(csvPath, style: const TextStyle(fontSize: 12)),

              const SizedBox(height: 12),

              const Text(
                'Lokasi JSON:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),

              SelectableText(jsonPath, style: const TextStyle(fontSize: 12)),
            ],
          ],
        ),
      ),
    );
  }
}
  