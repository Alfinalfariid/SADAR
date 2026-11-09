import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_background/flutter_background.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vibration/vibration.dart';

import 'overlay_bubble.dart';
import 'sadar_dashboard.dart';
import 'services/realtime_detection_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  LicenseRegistry.addLicense(() async* {
    yield LicenseEntryWithLineBreaks([
      'Sora',
      'Plus Jakarta Sans',
    ], await rootBundle.loadString('assets/fonts/FONT_LICENSES.txt'));
  });

  runApp(const SadarApp());
}

@pragma('vm:entry-point')
void overlayMain() {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: sadarTheme(),
      color: Colors.transparent,
      home: const DangerBubbleOverlay(),
    ),
  );
}

// BEGIN ALERT POLICY: pure Dart, independent of widgets/plugins.
class SadarAlert {
  const SadarAlert({
    required this.id,
    required this.label,
    required this.confidence,
    required this.latencyMs,
    required this.detectedAt,
  });

  final String id, label;
  final double confidence, latencyMs;
  final DateTime detectedAt;

  int get priority => label == 'siren' || label == 'vehicle_horn' ? 2 : 1;

  SadarAlert update(SadarAlert other) => SadarAlert(
    id: id,
    label: label,
    confidence: other.confidence,
    latencyMs: other.latencyMs,
    detectedAt: detectedAt,
  );
}

class SadarAlertPolicy {
  SadarAlertPolicy({required this.now});

  final Duration Function() now;
  final Map<String, Duration> _mutedUntil = {};
  final Map<String, SadarAlert> _pending = {};

  SadarAlert? current;
  bool enabled = false;

  Duration remaining(String label) {
    final delta = (_mutedUntil[label] ?? Duration.zero) - now();
    return delta.isNegative ? Duration.zero : delta;
  }

  bool receive(SadarAlert event) {
    if (!enabled ||
        event.label == 'background' ||
        event.label == '-' ||
        remaining(event.label) > Duration.zero) {
      return false;
    }

    final shown = current;

    if (shown == null) {
      current = event;
      return true;
    }

    if (shown.label == event.label) {
      current = shown.update(event);
      return true;
    }

    if (event.priority > shown.priority) {
      _pending[shown.label] = shown;
      _pending.remove(event.label);
      current = event;
      return true;
    }

    _pending[event.label] = _pending[event.label]?.update(event) ?? event;
    return false;
  }

  bool confirm(String id, String label) {
    if (current?.id != id || current?.label != label) {
      return false;
    }

    _mutedUntil[label] = now() + const Duration(minutes: 5);
    _pending.remove(label);

    final candidates =
        _pending.values
            .where((event) => remaining(event.label) == Duration.zero)
            .toList()
          ..sort((a, b) {
            final rank = b.priority.compareTo(a.priority);
            return rank != 0 ? rank : a.detectedAt.compareTo(b.detectedAt);
          });

    current = candidates.isEmpty ? null : candidates.first;

    if (current != null) {
      _pending.remove(current!.label);
    }

    return true;
  }

  void stop() {
    enabled = false;
    current = null;
    _pending.clear();
  }
}
// END ALERT POLICY

class SadarApp extends StatelessWidget {
  const SadarApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'SADAR',
    debugShowCheckedModeBanner: false,
    theme: sadarTheme(),
    home: const _Splash(),
  );
}

class _Splash extends StatefulWidget {
  const _Splash();

  @override
  State<_Splash> createState() => _SplashState();
}

class _SplashState extends State<_Splash> with SingleTickerProviderStateMixin {
  late final AnimationController _animation = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 750),
  )..forward();

  Timer? _timer;

  @override
  void initState() {
    super.initState();

    _timer = Timer(const Duration(milliseconds: 1300), () {
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute<void>(builder: (_) => const HomePage()),
        );
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _animation.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    body: Center(
      child: FadeTransition(
        opacity: _animation,
        child: ScaleTransition(
          scale: Tween<double>(
            begin: .92,
            end: 1,
          ).animate(CurvedAnimation(parent: _animation, curve: Curves.easeOut)),
          child: const Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SadarLogo(size: 124),
              SizedBox(height: 18),
              Text(
                'SADAR',
                style: TextStyle(
                  fontFamily: 'Sora',
                  fontSize: 30,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -1,
                ),
              ),
              SizedBox(height: 12),
              Text(
                'Kenali suara. Tingkatkan kewaspadaan.',
                style: TextStyle(fontSize: 12, color: SadarColors.muted),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _service = RealtimeDetectionService();
  final _prefs = SharedPreferencesAsync();
  final _clock = Stopwatch()..start();

  late final _policy = SadarAlertPolicy(now: () => _clock.elapsed);

  final _session = DateTime.now().microsecondsSinceEpoch.toString();

  StreamSubscription<RealtimeDetectionResult>? _results;
  Timer? _ackTimer;
  Future<void>? _syncTask;

  bool _syncNeeded = false, _readingAck = false, _disposed = false;
  bool isRunning = false, isLoading = false;
  bool backgroundActive = false, overlayActive = false;

  int processedWindows = 0, _serial = 0;

  String currentLabel = '-', status = '', _openedMode = '', _lastAck = '';

  double currentConfidence = 0, currentLatency = 0, currentRms = 0;

  @override
  void initState() {
    super.initState();

    _ackTimer = Timer.periodic(const Duration(milliseconds: 150), (_) {
      unawaited(_readConfirmation());
    });

    _requestSync();
  }

  void _requestSync() {
    if (_disposed) {
      return;
    }

    _syncNeeded = true;

    if (_syncTask != null) {
      return;
    }

    final task = _syncOverlay();
    _syncTask = task;

    unawaited(
      task.whenComplete(() {
        _syncTask = null;

        if (_syncNeeded && !_disposed) {
          _requestSync();
        }
      }),
    );
  }

  Future<void> _syncOverlay() async {
    while (_syncNeeded && !_disposed) {
      _syncNeeded = false;

      try {
        final alert = _policy.current;
        final mode = !isRunning
            ? 'off'
            : alert == null
            ? 'bubble'
            : 'danger';

        await _prefs.setString(
          SadarOverlayKeys.state,
          jsonEncode({
            'session': _session,
            'mode': mode,
            'id': alert?.id,
            'label': alert?.label ?? 'background',
            'confidence': alert?.confidence ?? 0.0,
            'latencyMs': alert?.latencyMs ?? 0.0,
            'detectedAt': alert?.detectedAt.millisecondsSinceEpoch,
          }),
        );

        final active = await FlutterOverlayWindow.isActive();

        if (mode == 'off') {
          if (active) {
            await FlutterOverlayWindow.closeOverlay();
          }

          _openedMode = '';
        } else if (!active || _openedMode != mode) {
          if (active) {
            await FlutterOverlayWindow.closeOverlay();
            await Future<void>.delayed(const Duration(milliseconds: 120));
          }

          if (!mounted || _disposed || !isRunning) {
            _syncNeeded = true;
            continue;
          }

          final danger = mode == 'danger';

          final view = View.of(context);
          final screenSize = view.display.size.isEmpty
              ? view.physicalSize
              : view.display.size;

          // Pada flutter_overlay_window 0.5.0, ukuran showOverlay
          // diteruskan langsung ke LayoutParams Android sebagai piksel fisik.
          final width = danger ? screenSize.width.round() : 180;
          final height = danger ? screenSize.height.round() : 180;

          if (kDebugMode && danger) {
            debugPrint(
              'SADAR overlay geometry: ${width}x${height}px, origin=(0,0)',
            );
          }

          await FlutterOverlayWindow.showOverlay(
            width: width,
            height: height,
            alignment: danger
                ? OverlayAlignment.topLeft
                : OverlayAlignment.centerRight,
            startPosition: danger ? const OverlayPosition(0, 0) : null,
            flag: OverlayFlag.defaultFlag,
            enableDrag: !danger,
            positionGravity: danger
                ? PositionGravity.none
                : PositionGravity.auto,
            overlayTitle: 'SADAR aktif',
            overlayContent: 'Memantau suara di sekitar.',
            visibility: NotificationVisibility.visibilityPublic,
          );

          _openedMode = mode;
        }

        if (mounted && overlayActive != (mode != 'off')) {
          setState(() => overlayActive = mode != 'off');
        }
      } catch (error) {
        debugPrint('SADAR overlay: $error');

        if (mounted) {
          setState(() => status = 'Gagal menampilkan peringatan: $error');
        }
      }
    }
  }

  Future<void> _readConfirmation() async {
    if (_readingAck || _disposed || !isRunning) {
      return;
    }

    _readingAck = true;

    try {
      final raw = await _prefs.getString(SadarOverlayKeys.confirmation);

      if (raw == null || raw == _lastAck) {
        return;
      }

      _lastAck = raw;
      final data = jsonDecode(raw) as Map<String, dynamic>;

      if (data['session'] != _session) {
        return;
      }

      if (!_policy.confirm(data['id'] as String, data['label'] as String)) {
        return;
      }

      await Vibration.cancel();
      _requestSync();

      if (_policy.current != null) {
        unawaited(_vibrate());
      }
    } catch (error) {
      debugPrint('SADAR confirmation: $error');
    } finally {
      _readingAck = false;
    }
  }

  Future<void> _vibrate() async {
    try {
      if (await Vibration.hasVibrator()) {
        await Vibration.vibrate(pattern: [0, 400, 150, 400]);
      }
    } catch (error) {
      debugPrint('SADAR vibration: $error');
    }
  }

  void _onResult(RealtimeDetectionResult result) {
    if (!mounted || !isRunning) {
      return;
    }

    final previousId = _policy.current?.id;

    final changed = _policy.receive(
      SadarAlert(
        id: '$_session-${++_serial}',
        label: result.label,
        confidence: result.confidence,
        latencyMs: result.latencyMs,
        detectedAt: DateTime.now(),
      ),
    );

    if (changed) {
      if (_policy.current?.id != previousId) {
        unawaited(_vibrate());
      }

      _requestSync();
    }

    setState(() {
      currentLabel = result.label;
      currentConfidence = result.confidence;
      currentLatency = result.latencyMs;
      currentRms = result.rms;
      processedWindows++;
    });
  }

  Future<void> startDetection() async {
    if (isRunning || isLoading) {
      return;
    }

    setState(() {
      isLoading = true;
      status = '';
    });

    try {
      if (!await FlutterOverlayWindow.isPermissionGranted()) {
        await FlutterOverlayWindow.requestPermission();

        if (!await FlutterOverlayWindow.isPermissionGranted()) {
          throw Exception('Izinkan SADAR tampil di atas aplikasi lain.');
        }
      }

      final ready = await FlutterBackground.initialize(
        androidConfig: const FlutterBackgroundAndroidConfig(
          notificationTitle: 'SADAR aktif',
          notificationText: 'Memantau suara bahaya di sekitar.',
          notificationImportance: AndroidNotificationImportance.normal,
          enableWifiLock: false,
        ),
      );

      if (!ready || !await FlutterBackground.enableBackgroundExecution()) {
        throw Exception('Layanan latar belakang gagal diaktifkan.');
      }

      backgroundActive = true;

      await _prefs.remove(SadarOverlayKeys.confirmation);
      _lastAck = '';

      await _service.start();
      _policy.enabled = true;

      if (!mounted) {
        return;
      }

      setState(() {
        isRunning = true;
        processedWindows = 0;
      });

      await _results?.cancel();

      _results = _service.results.listen(
        _onResult,
        onError: (Object error) {
          if (mounted) {
            setState(() => status = 'Gagal membaca suara: $error');
          }
        },
      );

      _requestSync();
    } catch (error) {
      await _results?.cancel();
      _results = null;

      await _service.stop();
      _policy.stop();
      isRunning = false;

      await _disableBackground();
      _requestSync();

      if (mounted) {
        setState(() => status = 'Gagal mengaktifkan SADAR: $error');
      }
    } finally {
      if (mounted) {
        setState(() => isLoading = false);
      }
    }
  }

  Future<void> _disableBackground() async {
    if (FlutterBackground.isBackgroundExecutionEnabled) {
      await FlutterBackground.disableBackgroundExecution();
    }

    backgroundActive = false;
  }

  Future<void> stopDetection() async {
    if (!isRunning || isLoading) {
      return;
    }

    setState(() {
      isLoading = true;
      isRunning = false;
      status = '';
    });

    _policy.stop();
    _requestSync();

    try {
      await _results?.cancel();
      _results = null;

      await _service.stop();
      await Vibration.cancel();
      await _syncTask;
      await _disableBackground();
      await _prefs.remove(SadarOverlayKeys.confirmation);

      if (mounted) {
        setState(() {
          currentLabel = '-';
          overlayActive = false;
        });
      }
    } catch (error) {
      if (mounted) {
        setState(() => status = 'Gagal menghentikan SADAR: $error');
      }
    } finally {
      if (mounted) {
        setState(() => isLoading = false);
      }
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _ackTimer?.cancel();
    _clock.stop();

    unawaited(_results?.cancel());
    unawaited(_service.dispose());

    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SadarDashboard(
    isRunning: isRunning,
    isLoading: isLoading,
    backgroundActive: backgroundActive,
    overlayActive: overlayActive,
    currentLabel: currentLabel,
    currentConfidence: currentConfidence,
    currentLatency: currentLatency,
    currentRms: currentRms,
    processedWindows: processedWindows,
    status: status,
    onStart: startDetection,
    onStop: stopDetection,
  );
}
