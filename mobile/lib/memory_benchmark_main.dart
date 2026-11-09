import 'package:flutter/material.dart';

import 'services/memory_benchmark_service.dart';

void main() {
  runApp(
    const MemoryBenchmarkApp(),
  );
}

class MemoryBenchmarkApp extends StatelessWidget {
  const MemoryBenchmarkApp({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'YAMNet Memory Benchmark',
      theme: ThemeData(
        useMaterial3: true,
      ),
      home: const MemoryBenchmarkPage(),
    );
  }
}

class MemoryBenchmarkPage extends StatefulWidget {
  const MemoryBenchmarkPage({
    super.key,
  });

  @override
  State<MemoryBenchmarkPage> createState() =>
      _MemoryBenchmarkPageState();
}

class _MemoryBenchmarkPageState
    extends State<MemoryBenchmarkPage> {
  final MemoryBenchmarkService _service =
      MemoryBenchmarkService();

  bool isLoading = false;

  String status =
      'Belum ada model yang dimuat.\n\n'
      'Ukur BASELINE RAM sekarang.';

  Future<void> loadFp32() async {
    setState(() {
      isLoading = true;
      status =
          'Loading YAMNet FP32...';
    });

    try {
      await _service.loadModel(
        modelPath:
            'assets/models/yamnet_embedding_fp32.tflite',
        modelName:
            'YAMNet FP32',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        status =
            'YAMNet FP32 SIAP\n\n'
            'Warmup 20x selesai.\n'
            'Sekarang ukur RAM dari laptop.';
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        status =
            'ERROR FP32:\n$e';
      });
    } finally {
      if (mounted) {
        setState(() {
          isLoading = false;
        });
      }
    }
  }

  Future<void> loadDynamicInt8() async {
    setState(() {
      isLoading = true;
      status =
          'Loading YAMNet Dynamic INT8...';
    });

    try {
      await _service.loadModel(
        modelPath:
            'assets/models/yamnet_embedding_dynamic_int8.tflite',
        modelName:
            'YAMNet Dynamic INT8',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        status =
            'YAMNet Dynamic INT8 SIAP\n\n'
            'Warmup 20x selesai.\n'
            'Sekarang ukur RAM dari laptop.';
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        status =
            'ERROR INT8:\n$e';
      });
    } finally {
      if (mounted) {
        setState(() {
          isLoading = false;
        });
      }
    }
  }

  void closeModel() {
    _service.closeModel();

    setState(() {
      status =
          'Model ditutup.\n\n'
          'Untuk pengukuran berikutnya '
          'lebih baik restart aplikasi.';
    });
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Memory Benchmark',
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding:
              const EdgeInsets.all(24),
          children: [
            const Icon(
              Icons.memory,
              size: 80,
            ),

            const SizedBox(
              height: 20,
            ),

            const Text(
              'YAMNet Memory Usage',
              textAlign:
                  TextAlign.center,
              style: TextStyle(
                fontSize: 25,
                fontWeight:
                    FontWeight.bold,
              ),
            ),

            const SizedBox(
              height: 20,
            ),

            Container(
              width:
                  double.infinity,
              padding:
                  const EdgeInsets.all(20),
              decoration:
                  BoxDecoration(
                borderRadius:
                    BorderRadius.circular(
                  16,
                ),
                color:
                    Theme.of(context)
                        .colorScheme
                        .surfaceContainerHighest,
              ),
              child: Text(
                status,
                textAlign:
                    TextAlign.center,
              ),
            ),

            const SizedBox(
              height: 25,
            ),

            if (isLoading)
              const Center(
                child:
                    CircularProgressIndicator(),
              ),

            const SizedBox(
              height: 15,
            ),

            ElevatedButton.icon(
              onPressed:
                  isLoading
                      ? null
                      : loadFp32,
              icon:
                  const Icon(
                Icons.memory,
              ),
              label:
                  const Text(
                'Load YAMNet FP32',
              ),
            ),

            const SizedBox(
              height: 12,
            ),

            ElevatedButton.icon(
              onPressed:
                  isLoading
                      ? null
                      : loadDynamicInt8,
              icon:
                  const Icon(
                Icons.compress,
              ),
              label:
                  const Text(
                'Load Dynamic INT8',
              ),
            ),

            const SizedBox(
              height: 12,
            ),

            OutlinedButton.icon(
              onPressed:
                  isLoading
                      ? null
                      : closeModel,
              icon:
                  const Icon(
                Icons.close,
              ),
              label:
                  const Text(
                'Close Model',
              ),
            ),
          ],
        ),
      ),
    );
  }
}