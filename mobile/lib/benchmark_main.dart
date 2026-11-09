import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'services/model_benchmark_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const BenchmarkApp());
}

class BenchmarkApp extends StatelessWidget {
  const BenchmarkApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SADAR YAMNet Benchmark',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const BenchmarkPage(),
    );
  }
}

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  final ModelBenchmarkService _service = ModelBenchmarkService();

  static const int _warmupRuns = 20;
  static const int _measuredRuns = 100;
  static const int _repetitions = 3;

  static const Map<String, String> _models = {
    'FP32': 'assets/models/yamnet_embedding_fp32.tflite',
    'Dynamic_INT8': 'assets/models/yamnet_embedding_dynamic_int8.tflite',
  };

  bool _running = false;
  bool _completed = false;

  String _status = 'Siap menjalankan benchmark.';
  String _csv = '';
  String _savedPath = '';
  String _summary = '';

  final List<Map<String, dynamic>> _records = [];

  Future<void> _runBenchmark() async {
    if (_running) {
      return;
    }

    setState(() {
      _running = true;
      _completed = false;
      _records.clear();

      _csv = '';
      _savedPath = '';
      _summary = '';

      _status = 'Menyiapkan benchmark...';
    });

    try {
      const csvHeader =
          'run,sequence,model,'
          'warmup_runs,measured_runs,'
          'mean_ms,median_ms,std_ms,'
          'min_ms,max_ms';

      final csvLines = <String>[csvHeader];

      var sequenceNumber = 0;

      for (var run = 1; run <= _repetitions; run++) {
        // Urutan dibalik pada run kedua
        // untuk mengurangi pengaruh urutan
        // pengujian terhadap hasil.

        final order = run.isEven
            ? <String>['Dynamic_INT8', 'FP32']
            : <String>['FP32', 'Dynamic_INT8'];

        for (final modelKey in order) {
          sequenceNumber++;

          if (!mounted) {
            return;
          }

          setState(() {
            _status =
                'Mengukur $modelKey\n'
                'Run $run/$_repetitions\n'
                'Pengujian $sequenceNumber/6';
          });

          // Beri kesempatan UI memperbarui
          // tampilan sebelum inferensi dimulai.

          await Future<void>.delayed(const Duration(milliseconds: 300));

          final result = await _service.benchmarkYamNet(
            modelPath: _models[modelKey]!,
            modelName: 'YAMNet $modelKey',
            warmupRuns: _warmupRuns,
            measuredRuns: _measuredRuns,
          );

          final record = <String, dynamic>{
            'run': run,
            'sequence': sequenceNumber,
            'model': modelKey,
            'warmup_runs': _warmupRuns,
            'measured_runs': _measuredRuns,
            'mean_ms': result.meanMs,
            'median_ms': result.medianMs,
            'std_ms': result.stdMs,
            'min_ms': result.minMs,
            'max_ms': result.maxMs,
          };

          _records.add(record);

          final csvRow = [
            run.toString(),
            sequenceNumber.toString(),
            modelKey,
            _warmupRuns.toString(),
            _measuredRuns.toString(),
            result.meanMs.toStringAsFixed(6),
            result.medianMs.toStringAsFixed(6),
            result.stdMs.toStringAsFixed(6),
            result.minMs.toStringAsFixed(6),
            result.maxMs.toStringAsFixed(6),
          ].join(',');

          csvLines.add(csvRow);

          debugPrint('SADAR_BENCHMARK_ROW:$csvRow');

          if (!mounted) {
            return;
          }

          setState(() {
            _status =
                'Selesai: $modelKey run $run\n'
                'Mean: ${result.meanMs.toStringAsFixed(3)} ms';
          });

          await Future<void>.delayed(const Duration(milliseconds: 500));
        }
      }

      _csv = '${csvLines.join('\n')}\n';

      double averageMean(String modelKey) {
        final values = _records
            .where((record) => record['model'] == modelKey)
            .map((record) => record['mean_ms'] as double)
            .toList();

        return values.reduce((a, b) => a + b) / values.length;
      }

      final fp32Mean = averageMean('FP32');
      final int8Mean = averageMean('Dynamic_INT8');

      final latencyReduction = (fp32Mean - int8Mean) / fp32Mean * 100.0;

      _summary =
          'YAMNet FP32\n'
          'Rata-rata mean: '
          '${fp32Mean.toStringAsFixed(3)} ms\n\n'
          'YAMNet Dynamic INT8\n'
          'Rata-rata mean: '
          '${int8Mean.toStringAsFixed(3)} ms\n\n'
          'Perubahan latensi:\n'
          '${latencyReduction.toStringAsFixed(2)}%\n\n'
          'Nilai positif berarti INT8 lebih cepat. '
          'Nilai negatif berarti INT8 lebih lambat.';

      // Simpan file di cache privat aplikasi.
      // File ini dapat disalin melalui ADB
      // ketika aplikasi berjalan sebagai
      // debug build.

      final outputDirectory = Directory.systemTemp;

      final csvFile = File(
        '${outputDirectory.path}/'
        'sadar_yamnet_latency_summary.csv',
      );

      await csvFile.writeAsString(_csv, flush: true);

      final metadata = <String, dynamic>{
        'experiment': 'SADAR YAMNet Latency Benchmark',
        'created_at': DateTime.now().toIso8601String(),
        'fp32_model': _models['FP32'],
        'dynamic_int8_model': _models['Dynamic_INT8'],
        'repetitions_per_model': _repetitions,
        'warmup_runs_per_repetition': _warmupRuns,
        'measured_runs_per_repetition': _measuredRuns,
        'total_measured_inferences': _repetitions * _measuredRuns * 2,
        'input_samples': 15600,
        'input_description':
            'Synthetic 440 Hz sine waveform, '
            '16000 Hz sample rate, float32',
        'measurement_scope':
            'Interpreter.run() only; excludes '
            'interpreter loading, tensor allocation, '
            'audio recording and UI alert',
        'fresh_interpreter_per_repetition': true,
        'model_order':
            'Run 1 FP32-INT8; '
            'Run 2 INT8-FP32; '
            'Run 3 FP32-INT8',
        'fp32_average_mean_ms': fp32Mean,
        'int8_average_mean_ms': int8Mean,
        'latency_reduction_percent': latencyReduction,
        'results': _records,
      };

      final metadataFile = File(
        '${outputDirectory.path}/'
        'sadar_yamnet_latency_metadata.json',
      );

      await metadataFile.writeAsString(
        const JsonEncoder.withIndent('  ').convert(metadata),
        flush: true,
      );

      debugPrint(
        'SADAR_BENCHMARK_CSV_PATH:'
        '${csvFile.path}',
      );

      debugPrint(
        'SADAR_BENCHMARK_JSON_PATH:'
        '${metadataFile.path}',
      );

      debugPrint('SADAR_BENCHMARK_COMPLETE:6');

      if (!mounted) {
        return;
      }

      setState(() {
        _savedPath = csvFile.path;
        _completed = true;
        _status = 'Benchmark selesai dan CSV tersimpan.';
      });
    } catch (error, stackTrace) {
      debugPrint('Benchmark error: $error');

      debugPrintStack(stackTrace: stackTrace);

      if (mounted) {
        setState(() {
          _status = 'Benchmark gagal:\n$error';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
        });
      }
    }
  }

  Future<void> _copyCsv() async {
    if (_csv.isEmpty) {
      return;
    }

    await Clipboard.setData(ClipboardData(text: _csv));

    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Hasil CSV disalin ke clipboard.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SADAR Benchmark'), centerTitle: true),
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

            const SizedBox(height: 12),

            const Text(
              'Benchmark inferensi on-device '
              'menggunakan input yang sama.',
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 20),

            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text('Konfigurasi pengujian'),
                    SizedBox(height: 8),
                    Text('• Tiga run per model'),
                    Text('• 20 warmup per run'),
                    Text('• 100 pengukuran per run'),
                    Text('• Input 15.600 sampel'),
                    Text('• Sinyal sinus 440 Hz'),
                    Text('• Model dimuat ulang setiap run'),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            SizedBox(
              height: 52,
              child: FilledButton.icon(
                onPressed: _running ? null : _runBenchmark,
                icon: Icon(
                  _running
                      ? Icons.hourglass_top_rounded
                      : Icons.play_arrow_rounded,
                ),
                label: Text(_running ? 'SEDANG MENGUJI' : 'MULAI BENCHMARK'),
              ),
            ),

            const SizedBox(height: 16),

            if (_running) const Center(child: CircularProgressIndicator()),

            const SizedBox(height: 12),

            Text(
              _status,
              textAlign: TextAlign.center,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),

            if (_completed) ...[
              const SizedBox(height: 20),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: SelectableText(
                    _summary,
                    style: const TextStyle(fontSize: 15),
                  ),
                ),
              ),

              const SizedBox(height: 12),

              OutlinedButton.icon(
                onPressed: _copyCsv,
                icon: const Icon(Icons.copy_rounded),
                label: const Text('SALIN HASIL CSV'),
              ),

              const SizedBox(height: 12),

              const Text(
                'Lokasi CSV pada Android:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),

              const SizedBox(height: 5),

              SelectableText(_savedPath, style: const TextStyle(fontSize: 12)),

              const SizedBox(height: 8),

              const Text(
                'Hasil disimpan di cache aplikasi. '
                'Salin ke laptop sebelum membersihkan '
                'data atau cache aplikasi.',
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
