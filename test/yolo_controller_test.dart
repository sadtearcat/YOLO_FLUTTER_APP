// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOViewController', () {
    late YOLOViewController controller;
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      controller = YOLOViewController();
      log.clear();

      // Note: Cannot test _init directly as it's private
      // Controller methods will handle missing channel gracefully
    });

    tearDown() {
      // No cleanup needed since we don't mock channels
    }

    test('default values are set correctly', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps values', () async {
      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setConfidenceThreshold(-0.5);
      expect(controller.confidenceThreshold, 0.0);
    });

    test('setIoUThreshold clamps values', () async {
      await controller.setIoUThreshold(1.2);
      expect(controller.iouThreshold, 1.0);

      await controller.setIoUThreshold(-0.1);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps values', () async {
      await controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);

      await controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });

    test('setThresholds updates multiple values at once', () async {
      await controller.setThresholds(
        confidenceThreshold: 0.8,
        iouThreshold: 0.6,
        numItemsThreshold: 50,
      );

      expect(controller.confidenceThreshold, 0.8);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 50);
    });

    test('setThresholds updates only provided values', () async {
      await controller.setThresholds(confidenceThreshold: 0.7);

      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.45); // unchanged
      expect(controller.numItemsThreshold, 30); // unchanged
    });

    test('switchCamera handles uninitialized channel gracefully', () async {
      // Should not throw when no method channel is set
      expect(() => controller.switchCamera(), returnsNormally);
    });

    test(
      'methods handle platform channel not initialized gracefully',
      () async {
        final uninitializedController = YOLOViewController();

        // Should not throw, just log warning
        expect(
          () => uninitializedController.setConfidenceThreshold(0.8),
          returnsNormally,
        );
        expect(() => uninitializedController.switchCamera(), returnsNormally);
      },
    );
  });

  group('YOLOViewController useGpu Tests', () {
    late YOLOViewController controller;
    const MethodChannel testChannel = MethodChannel('test_yolo_controller');
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      controller = YOLOViewController();
      log.clear();

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (MethodCall methodCall) async {
            log.add(methodCall);
            return null;
          });

      controller.init(testChannel, 1);
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, null);
    });

    test('switchModel with useGpu=false', () async {
      await controller.switchModel('test_model.tflite', YOLOTask.detect, useGpu: false);

      expect(log.any((call) => call.method == 'setModel'), isTrue);
      final setModelCall = log.firstWhere((call) => call.method == 'setModel');
      expect(setModelCall.arguments['modelPath'], 'test_model.tflite');
      expect(setModelCall.arguments['task'], 'detect');
      expect(setModelCall.arguments['useGpu'], false);
    });

    test('switchModel with useGpu=true (explicit)', () async {
      await controller.switchModel('test_model.tflite', YOLOTask.segment, useGpu: true);

      expect(log.any((call) => call.method == 'setModel'), isTrue);
      final setModelCall = log.firstWhere((call) => call.method == 'setModel');
      expect(setModelCall.arguments['useGpu'], true);
    });

    test('switchModel uses default useGpu=true when not specified', () async {
      await controller.switchModel('test_model.tflite', YOLOTask.pose);

      expect(log.any((call) => call.method == 'setModel'), isTrue);
      final setModelCall = log.firstWhere((call) => call.method == 'setModel');
      expect(setModelCall.arguments['useGpu'], true);
    });

    test('switchModel with different tasks and useGpu values', () async {
      // Test multiple combinations
      await controller.switchModel('model1.tflite', YOLOTask.detect, useGpu: false);
      await controller.switchModel('model2.tflite', YOLOTask.segment, useGpu: true);
      await controller.switchModel('model3.tflite', YOLOTask.classify); // default

      expect(log.where((call) => call.method == 'setModel').length, 3);
      
      final calls = log.where((call) => call.method == 'setModel').toList();
      expect(calls[0].arguments['useGpu'], false);
      expect(calls[1].arguments['useGpu'], true);
      expect(calls[2].arguments['useGpu'], true); // default
    });

    test('switchModel without initialization handles gracefully', () async {
      final uninitializedController = YOLOViewController();

      expect(
        () => uninitializedController.switchModel('model.tflite', YOLOTask.detect, useGpu: false),
        returnsNormally,
      );
    });
  });
}
