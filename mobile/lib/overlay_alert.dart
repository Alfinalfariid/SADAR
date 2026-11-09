import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

class DangerOverlay extends StatefulWidget {
  const DangerOverlay({
    super.key,
  });

  @override
  State<DangerOverlay> createState() =>
      _DangerOverlayState();
}

class _DangerOverlayState
    extends State<DangerOverlay> {
  StreamSubscription<dynamic>?
      _overlaySubscription;

  String label =
      'Bahaya Terdeteksi';

  double confidence =
      0.0;

  @override
  void initState() {
    super.initState();

    _overlaySubscription =
        FlutterOverlayWindow
            .overlayListener
            .listen(
      (event) {
        try {
          final data =
              jsonDecode(
            event.toString(),
          );

          if (!mounted) {
            return;
          }

          setState(() {
            label =
                data['label'] ??
                    'Bahaya Terdeteksi';

            confidence =
                (data['confidence'] ??
                        0.0)
                    .toDouble();
          });
        } catch (e) {
          debugPrint(
            'Overlay data error: $e',
          );
        }
      },
    );
  }

  String formatLabel(
    String value,
  ) {
    switch (value) {
      case 'alarm':
        return 'ALARM';

      case 'glass_breaking':
        return 'GLASS BREAKING';

      case 'scream':
        return 'SCREAM';

      case 'siren':
        return 'SIREN';

      case 'vehicle_horn':
        return 'VEHICLE HORN';

      default:
        return value
            .replaceAll(
              '_',
              ' ',
            )
            .toUpperCase();
    }
  }

  @override
  void dispose() {
    _overlaySubscription
        ?.cancel();

    super.dispose();
  }

  @override
  Widget build(
    BuildContext context,
  ) {
    return Material(
      color: Colors.transparent,
      child: SafeArea(
        child: Padding(
          padding:
              const EdgeInsets.all(
            12,
          ),
          child: Container(
            padding:
                const EdgeInsets.all(
              18,
            ),
            decoration:
                BoxDecoration(
              color:
                  const Color(
                0xFFFDECEC,
              ),
              borderRadius:
                  BorderRadius.circular(
                20,
              ),
              border:
                  Border.all(
                color:
                    Colors.red,
                width:
                    2,
              ),
              boxShadow: const [
                BoxShadow(
                  color:
                      Color(
                    0x33000000,
                  ),
                  blurRadius:
                      12,
                  offset:
                      Offset(
                    0,
                    5,
                  ),
                ),
              ],
            ),
            child: Row(
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration:
                      const BoxDecoration(
                    color:
                        Colors.red,
                    shape:
                        BoxShape.circle,
                  ),
                  child:
                      const Icon(
                    Icons
                        .warning_amber_rounded,
                    color:
                        Colors.white,
                    size:
                        32,
                  ),
                ),

                const SizedBox(
                  width: 14,
                ),

                Expanded(
                  child: Column(
                    mainAxisSize:
                        MainAxisSize.min,
                    crossAxisAlignment:
                        CrossAxisAlignment
                            .start,
                    children: [
                      const Text(
                        'BAHAYA TERDETEKSI',
                        style:
                            TextStyle(
                          color:
                              Colors.red,
                          fontSize:
                              14,
                          fontWeight:
                              FontWeight
                                  .bold,
                        ),
                      ),

                      const SizedBox(
                        height: 4,
                      ),

                      Text(
                        formatLabel(
                          label,
                        ),
                        maxLines:
                            1,
                        overflow:
                            TextOverflow
                                .ellipsis,
                        style:
                            const TextStyle(
                          color:
                              Colors.black87,
                          fontSize:
                              20,
                          fontWeight:
                              FontWeight
                                  .bold,
                        ),
                      ),

                      const SizedBox(
                        height: 4,
                      ),

                      Text(
                        'Confidence '
                        '${(confidence * 100).toStringAsFixed(1)}%',
                        style:
                            const TextStyle(
                          color:
                              Colors.black54,
                          fontSize:
                              13,
                        ),
                      ),
                    ],
                  ),
                ),

                IconButton(
                  onPressed:
                      () async {
                    await FlutterOverlayWindow
                        .closeOverlay();
                  },
                  icon:
                      const Icon(
                    Icons.close,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}