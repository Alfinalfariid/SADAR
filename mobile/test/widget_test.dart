import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import '../lib/sadar_dashboard.dart';

void main() {
  testWidgets('SADAR menampilkan dashboard dan tombol aktivasi', (
    WidgetTester tester,
  ) async {
    bool startPressed = false;
    bool stopPressed = false;

    await tester.pumpWidget(
      MaterialApp(
        home: SadarDashboard(
          isRunning: false,
          isLoading: false,
          backgroundActive: false,
          overlayActive: false,
          currentLabel: '-',
          currentConfidence: 0.0,
          currentLatency: 0.0,
          currentRms: 0.0,
          processedWindows: 0,
          status: 'SADAR belum aktif',
          onStart: () {
            startPressed = true;
          },
          onStop: () {
            stopPressed = true;
          },
        ),
      ),
    );

    expect(find.text('SADAR'), findsOneWidget);

    expect(find.text('AKTIFKAN SADAR'), findsOneWidget);

    expect(find.text('Siap menemanimu'), findsOneWidget);

    final activateButton = find.text('AKTIFKAN SADAR');

    await tester.ensureVisible(activateButton);
    await tester.tap(activateButton);
    await tester.pump();

    expect(startPressed, isTrue);
    expect(stopPressed, isFalse);
  });
}
