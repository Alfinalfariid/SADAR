import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'sadar_dashboard.dart';

class DangerBubbleOverlay extends StatefulWidget {
  const DangerBubbleOverlay({super.key});

  @override
  State<DangerBubbleOverlay> createState() => _DangerBubbleOverlayState();
}

class _DangerBubbleOverlayState extends State<DangerBubbleOverlay> {
  final _prefs = SharedPreferencesAsync();

  Timer? _timer;
  bool _reading = false, _confirming = false;

  String _raw = '',
      _session = '',
      _mode = 'off',
      _id = '',
      _label = 'background';

  double _confidence = 0, _latency = 0;
  DateTime _detectedAt = DateTime.now();

  @override
  void initState() {
    super.initState();
    unawaited(_read());

    _timer = Timer.periodic(
      const Duration(milliseconds: 150),
      (_) => unawaited(_read()),
    );
  }

  Future<void> _read() async {
    if (_reading) {
      return;
    }

    _reading = true;

    try {
      final raw = await _prefs.getString(SadarOverlayKeys.state);

      if (raw == null || raw == _raw) {
        return;
      }

      final data = jsonDecode(raw) as Map<String, dynamic>;

      if (!mounted) {
        return;
      }

      setState(() {
        final newId = data['id'] as String? ?? '';

        if (newId != _id || data['session'] != _session) {
          _confirming = false;
        }

        _raw = raw;
        _id = newId;
        _session = data['session'] as String? ?? '';
        _mode = data['mode'] as String? ?? 'off';
        _label = data['label'] as String? ?? 'background';
        _confidence = (data['confidence'] as num?)?.toDouble() ?? 0;
        _latency = (data['latencyMs'] as num?)?.toDouble() ?? 0;
        _detectedAt = DateTime.fromMillisecondsSinceEpoch(
          (data['detectedAt'] as num?)?.toInt() ??
              DateTime.now().millisecondsSinceEpoch,
        );
      });
    } catch (error) {
      debugPrint('SADAR overlay state: $error');
    } finally {
      _reading = false;
    }
  }

  Future<void> _confirm() async {
    if (_confirming || _id.isEmpty) {
      return;
    }

    setState(() => _confirming = true);

    try {
      await _prefs.setString(
        SadarOverlayKeys.confirmation,
        jsonEncode({'session': _session, 'id': _id, 'label': _label}),
      );
    } catch (error) {
      if (mounted) {
        setState(() => _confirming = false);
        ScaffoldMessenger.maybeOf(context)?.showSnackBar(
          const SnackBar(content: Text('Konfirmasi gagal. Coba lagi.')),
        );
      }
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Widget _bubble() => const Center(
    child: DecoratedBox(
      decoration: BoxDecoration(
        color: Color(0xE6333333),
        borderRadius: BorderRadius.all(Radius.circular(18)),
      ),
      child: Padding(
        padding: EdgeInsets.all(7),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.all(Radius.circular(12)),
          ),
          child: Padding(
            padding: EdgeInsets.all(3),
            child: SadarLogo(size: 36),
          ),
        ),
      ),
    ),
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: _mode == 'danger'
        ? const Color(0x55000000)
        : Colors.transparent,
    resizeToAvoidBottomInset: false,
    body: SizedBox.expand(
      child: _mode == 'off'
          ? const SizedBox.shrink()
          : LayoutBuilder(
              builder: (context, constraints) {
                if (_mode != 'danger' ||
                    constraints.maxWidth < 230 ||
                    constraints.maxHeight < 250) {
                  return _bubble();
                }

                return Padding(
                  padding: const EdgeInsets.all(20),
                  child: Center(
                    child: FittedBox(
                      fit: BoxFit.scaleDown,
                      child: SadarAlertCard(
                        label: _label,
                        confidence: _confidence,
                        latencyMs: _latency,
                        detectedAt: _detectedAt,
                        confirming: _confirming,
                        onConfirm: _confirm,
                      ),
                    ),
                  ),
                );
              },
            ),
    ),
  );
}
