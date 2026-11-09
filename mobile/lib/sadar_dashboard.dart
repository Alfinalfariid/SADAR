import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

class SadarColors {
  static const red = Color(0xFFB3262D);
  static const wave = Color(0xFF8D2028);
  static const ink = Color(0xFF242122);
  static const muted = Color(0xFF686466);
  static const yellow = Color(0xFFEFC541);
  static const green = Color(0xFF21834A);
}

class SadarOverlayKeys {
  static const state = 'sadar_overlay_state_v2';
  static const confirmation = 'sadar_overlay_confirmation_v2';
}

ThemeData sadarTheme() => ThemeData(
  useMaterial3: true,
  brightness: Brightness.light,
  scaffoldBackgroundColor: Colors.white,
  fontFamily: 'PlusJakartaSans',
  colorScheme: ColorScheme.fromSeed(
    seedColor: SadarColors.red,
    brightness: Brightness.light,
    surface: Colors.white,
  ),
  textTheme: ThemeData.light().textTheme.apply(
    fontFamily: 'PlusJakartaSans',
    bodyColor: SadarColors.ink,
    displayColor: SadarColors.ink,
  ),
);

class SadarLogo extends StatelessWidget {
  const SadarLogo({super.key, this.size = 40});

  final double size;

  @override
  Widget build(BuildContext context) => Image.asset(
    'assets/images/sadar_logo.png',
    width: size,
    height: size,
    fit: BoxFit.contain,
    semanticLabel: 'Logo SADAR',
    errorBuilder: (_, error, stack) =>
        Icon(Icons.hearing_rounded, size: size * .8, color: SadarColors.red),
  );
}

class SadarSound {
  static String name(String label) => switch (label) {
    'siren' => 'Sirine',
    'vehicle_horn' => 'Klakson',
    'alarm' => 'Alarm',
    'scream' => 'Teriakan',
    'glass_breaking' => 'Kaca pecah',
    _ => 'Suara latar',
  };

  static bool high(String label) => label == 'siren' || label == 'vehicle_horn';

  static IconData icon(String label) => switch (label) {
    'siren' => Icons.emergency_share_rounded,
    'vehicle_horn' => Icons.directions_car_filled_rounded,
    'alarm' => Icons.notifications_active_rounded,
    'scream' => Icons.record_voice_over_rounded,
    'glass_breaking' => Icons.broken_image_rounded,
    _ => Icons.graphic_eq_rounded,
  };

  static Color color(String label) => high(label)
      ? SadarColors.red
      : label == 'background'
      ? SadarColors.green
      : const Color(0xFFA37500);

  static String message(String label) => switch (label) {
    'siren' => 'Perhatikan sekitar. Mungkin ada kendaraan darurat.',
    'vehicle_horn' => 'Waspadai kendaraan di sekitar dan perhatikan jalan.',
    'alarm' => 'Periksa sumber alarm dan kondisi di sekitarmu.',
    'scream' => 'Perhatikan sekitar dan periksa sumber teriakan.',
    'glass_breaking' => 'Waspadai kemungkinan pecahan kaca di sekitar.',
    _ => 'Perhatikan lingkungan di sekitarmu.',
  };
}

class SadarDashboard extends StatefulWidget {
  const SadarDashboard({
    super.key,
    required this.isRunning,
    required this.isLoading,
    required this.backgroundActive,
    required this.overlayActive,
    required this.currentLabel,
    required this.currentConfidence,
    required this.currentLatency,
    required this.currentRms,
    required this.processedWindows,
    required this.status,
    required this.onStart,
    required this.onStop,
  });

  final bool isRunning, isLoading, backgroundActive, overlayActive;
  final String currentLabel, status;
  final double currentConfidence, currentLatency, currentRms;
  final int processedWindows;
  final VoidCallback onStart, onStop;

  @override
  State<SadarDashboard> createState() => _SadarDashboardState();
}

class _SadarDashboardState extends State<SadarDashboard>
    with SingleTickerProviderStateMixin {
  late final AnimationController _animation = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  );

  final _elapsed = Stopwatch();
  Timer? _timer;

  @override
  void initState() {
    super.initState();

    if (widget.isRunning) {
      _animation.repeat();
      _elapsed.start();
    }

    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted && widget.isRunning) {
        setState(() {});
      }
    });
  }

  @override
  void didUpdateWidget(covariant SadarDashboard oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (oldWidget.isRunning != widget.isRunning) {
      if (widget.isRunning) {
        _animation.repeat();
        _elapsed
          ..reset()
          ..start();
      } else {
        _animation.stop();
        _elapsed.stop();
      }
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _animation.dispose();
    super.dispose();
  }

  String get _time {
    final seconds = _elapsed.elapsed.inSeconds;

    return '${(seconds ~/ 60).toString().padLeft(2, '0')}:'
        '${(seconds % 60).toString().padLeft(2, '0')}';
  }

  void _help() => showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.white,
    showDragHandle: true,
    builder: (context) => SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 28),
        child: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Cara menggunakan SADAR',
              style: TextStyle(
                fontFamily: 'Sora',
                fontSize: 22,
                fontWeight: FontWeight.w700,
              ),
            ),
            SizedBox(height: 20),
            Text(
              '1. Aktifkan deteksi',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            Text(
              'Izinkan mikrofon, notifikasi, dan tampilan di atas aplikasi lain.',
            ),
            SizedBox(height: 16),
            Text(
              '2. Perhatikan peringatan',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            Text('Baca jenis suara, lalu periksa lingkungan sekitarmu.'),
            SizedBox(height: 16),
            Text(
              '3. Konfirmasi',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            Text(
              'Tekan “Saya sudah mengetahui”. Peringatan untuk jenis suara itu '
              'dijeda 5 menit. Deteksi tetap berjalan dan jenis lain tetap dapat '
              'memberi peringatan.',
            ),
            SizedBox(height: 16),
            Text(
              'Warna prioritas',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            Text(
              'Merah: sirine dan klakson. Kuning: alarm, teriakan, dan kaca pecah. '
              'Hijau: suara latar, tanpa popup bahaya.',
            ),
          ],
        ),
      ),
    ),
  );

  Widget _header() => Row(
    children: [
      const SadarLogo(size: 39),
      const SizedBox(width: 3),
      const Text(
        'SADAR',
        style: TextStyle(
          fontFamily: 'Sora',
          fontSize: 23,
          fontWeight: FontWeight.w700,
          letterSpacing: -1,
        ),
      ),
      const Spacer(),
      IconButton(
        onPressed: _help,
        tooltip: 'Bantuan penggunaan',
        icon: const Icon(
          Icons.help_outline_rounded,
          size: 24,
          color: SadarColors.ink,
        ),
      ),
    ],
  );

  Widget _intro() => const Column(
    mainAxisSize: MainAxisSize.min,
    children: [
      Text.rich(
        TextSpan(
          children: [
            TextSpan(text: 'Lebih sadar.\nLebih '),
            TextSpan(
              text: 'waspada.',
              style: TextStyle(color: SadarColors.red),
            ),
          ],
        ),
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 32,
          fontWeight: FontWeight.w700,
          letterSpacing: -1.3,
          height: 1.16,
          color: SadarColors.ink,
        ),
      ),
      SizedBox(height: 12),
      Text(
        'Kenali suara penting di sekitar,\nbahkan saat membuka aplikasi lain.',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 13, height: 1.7, color: SadarColors.muted),
      ),
    ],
  );

  Widget _hero() => Center(
    child: FittedBox(
      fit: BoxFit.scaleDown,
      child: AnimatedBuilder(
        animation: _animation,
        builder: (context, child) => SizedBox(
          width: 250,
          height: 250,
          child: CustomPaint(
            painter: _SoundWaves(_animation.value),
            child: child,
          ),
        ),
        child: const Center(child: SadarLogo(size: 124)),
      ),
    ),
  );

  Widget _controls() => Column(
    mainAxisSize: MainAxisSize.min,
    children: [
      AnimatedBuilder(
        animation: _animation,
        builder: (context, child) => Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.mic_rounded,
              size: 23,
              color: widget.isRunning ? SadarColors.red : Colors.grey,
            ),
            const SizedBox(width: 4),
            SizedBox(
              width: 20,
              height: 20,
              child: CustomPaint(
                painter: _MicBars(_animation.value, widget.isRunning),
              ),
            ),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                widget.isRunning ? 'Mikrofon aktif' : 'Mikrofon nonaktif',
                style: const TextStyle(fontSize: 13),
              ),
            ),
          ],
        ),
      ),
      const SizedBox(height: 20),
      SizedBox(
        width: double.infinity,
        height: 54,
        child: FilledButton.icon(
          onPressed: widget.isLoading
              ? null
              : widget.isRunning
              ? widget.onStop
              : widget.onStart,
          style: FilledButton.styleFrom(
            backgroundColor: SadarColors.red,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
          ),
          icon: widget.isLoading
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white,
                  ),
                )
              : const Icon(Icons.power_settings_new_rounded, size: 21),
          label: Text(
            widget.isLoading
                ? 'Memproses…'
                : widget.isRunning
                ? 'Hentikan deteksi'
                : 'Aktifkan deteksi',
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
        ),
      ),
      const SizedBox(height: 16),
      Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.access_time_rounded,
            size: 15,
            color: SadarColors.muted,
          ),
          const SizedBox(width: 7),
          Flexible(
            child: Text(
              widget.isRunning ? 'Waktu aktif' : 'Durasi sesi terakhir',
              style: const TextStyle(fontSize: 12, color: SadarColors.muted),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            _time,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
          ),
        ],
      ),
      if (widget.status.isNotEmpty)
        Padding(
          padding: const EdgeInsets.only(top: 12),
          child: Text(
            widget.status,
            textAlign: TextAlign.center,
            style: const TextStyle(color: SadarColors.red, fontSize: 12),
          ),
        ),
    ],
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: Colors.white,
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(22, 10, 22, 20),
            child: Column(
              children: [
                _header(),
                const SizedBox(height: 10),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final textScale = math.max(
                        1.0,
                        MediaQuery.textScalerOf(context).scale(13) / 13,
                      );
                      final minHeight = 520.0 * textScale;
                      final bodyHeight = math.max(
                        constraints.maxHeight,
                        minHeight,
                      );

                      return FittedBox(
                        fit: BoxFit.scaleDown,
                        alignment: Alignment.topCenter,
                        child: SizedBox(
                          width: constraints.maxWidth,
                          height: bodyHeight,
                          child: Column(
                            children: [
                              const SizedBox(height: 18),
                              _intro(),
                              Expanded(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 12,
                                  ),
                                  child: _hero(),
                                ),
                              ),
                              _controls(),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    ),
  );
}

class _SoundWaves extends CustomPainter {
  _SoundWaves(this.phase);

  final double phase;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = SadarColors.wave
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;

    for (var i = 0; i < 2; i++) {
      final t = (phase + i * .5) % 1;
      final radius = 76 * (.92 + .505 * t);
      final rect = Rect.fromCircle(
        center: size.center(Offset.zero),
        radius: radius,
      );

      canvas.drawArc(rect, -math.pi / 4, math.pi / 2, false, paint);
      canvas.drawArc(rect, 3 * math.pi / 4, math.pi / 2, false, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _SoundWaves old) => old.phase != phase;
}

class _MicBars extends CustomPainter {
  _MicBars(this.phase, this.active);

  final double phase;
  final bool active;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = active ? SadarColors.red : Colors.grey
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    for (var i = 0; i < 3; i++) {
      final h = active
          ? 5 + 11 * (.5 + .5 * math.sin(phase * math.pi * 6 + i * 1.8))
          : 5.0;
      final x = 3.0 + i * 6;

      canvas.drawLine(
        Offset(x, (size.height - h) / 2),
        Offset(x, (size.height + h) / 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _MicBars old) =>
      old.phase != phase || old.active != active;
}

class SadarAlertCard extends StatelessWidget {
  const SadarAlertCard({
    super.key,
    required this.label,
    required this.confidence,
    required this.latencyMs,
    required this.detectedAt,
    required this.confirming,
    required this.onConfirm,
  });

  final String label;
  final double confidence, latencyMs;
  final DateTime detectedAt;
  final bool confirming;
  final VoidCallback onConfirm;

  String get _time =>
      '${detectedAt.hour.toString().padLeft(2, '0')}:'
      '${detectedAt.minute.toString().padLeft(2, '0')}:'
      '${detectedAt.second.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final high = SadarSound.high(label);
    final color = SadarSound.color(label);
    final score = confidence.clamp(0.0, 1.0).toDouble();

    return Container(
      width: 320,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: const [BoxShadow(color: Color(0x33000000), blurRadius: 28)],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Text(
                'SADAR',
                style: TextStyle(
                  fontFamily: 'Sora',
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 7,
                ),
                decoration: BoxDecoration(
                  color: high ? SadarColors.red : SadarColors.yellow,
                  borderRadius: BorderRadius.circular(7),
                ),
                child: Text(
                  high ? 'Prioritas tinggi' : 'Prioritas sedang',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: high ? Colors.white : const Color(0xFF493700),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          const Text(
            'SUARA TERDETEKSI',
            style: TextStyle(
              fontSize: 10,
              letterSpacing: 1.2,
              fontWeight: FontWeight.w600,
              color: SadarColors.muted,
            ),
          ),
          const SizedBox(height: 8),
          if (label == 'siren')
            const SizedBox(
              width: 110,
              height: 110,
              child: CustomPaint(painter: _SirenPainter()),
            )
          else
            Icon(SadarSound.icon(label), size: 110, color: color),
          const SizedBox(height: 3),
          Text(
            SadarSound.name(label),
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontFamily: 'Sora',
              fontSize: 28,
              fontWeight: FontWeight.w700,
              letterSpacing: -1,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            SadarSound.message(label),
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 12,
              height: 1.6,
              color: SadarColors.muted,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Confidence AI', style: TextStyle(fontSize: 12)),
              Text(
                '${(score * 100).toStringAsFixed(1)}%',
                style: const TextStyle(
                  fontSize: 23,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 7),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: score,
              minHeight: 5,
              color: color,
              backgroundColor: const Color(0xFFEAEAEA),
            ),
          ),
          const SizedBox(height: 13),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                _time,
                style: const TextStyle(fontSize: 11, color: SadarColors.muted),
              ),
              Text(
                'Inferensi ${latencyMs.toStringAsFixed(1)} ms',
                style: const TextStyle(fontSize: 11, color: SadarColors.muted),
              ),
            ],
          ),
          const SizedBox(height: 18),
          SizedBox(
            width: double.infinity,
            height: 50,
            child: FilledButton.icon(
              onPressed: confirming ? null : onConfirm,
              style: FilledButton.styleFrom(
                backgroundColor: SadarColors.red,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              icon: const Icon(Icons.check_rounded, size: 19),
              label: Text(
                confirming ? 'Memproses…' : 'Saya sudah mengetahui',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Jeda 5 menit untuk suara ini',
            style: TextStyle(fontSize: 10, color: SadarColors.muted),
          ),
        ],
      ),
    );
  }
}

class _SirenPainter extends CustomPainter {
  const _SirenPainter();

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / 110, size.height / 110);

    final paint = Paint()
      ..color = SadarColors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final dome = Path()
      ..moveTo(29, 74)
      ..lineTo(29, 51)
      ..cubicTo(29, 18, 81, 18, 81, 51)
      ..lineTo(81, 74);

    canvas.drawPath(dome, paint);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        const Rect.fromLTWH(21, 74, 68, 16),
        const Radius.circular(6),
      ),
      paint,
    );

    canvas.drawLine(const Offset(55, 9), const Offset(55, 19), paint);
    canvas.drawLine(const Offset(15, 27), const Offset(23, 35), paint);
    canvas.drawLine(const Offset(95, 27), const Offset(87, 35), paint);

    final highlight = Path()
      ..moveTo(42, 55)
      ..quadraticBezierTo(42, 43, 50, 41);

    canvas.drawPath(highlight, paint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _SirenPainter oldDelegate) => false;
}
