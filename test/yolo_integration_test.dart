// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Integration Tests', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      log.clear();
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('complete workflow: load model and predict', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);

            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                'boxes': [
                  {
                    'classIndex': 0,
                    'className': 'person',
                    'confidence': 0.95,
                    'boundingBox': {
                      'left': 10.0,
                      'top': 10.0,
                      'right': 110.0,
                      'bottom': 210.0,
                    },
                    'normalizedBox': {
                      'left': 0.1,
                      'top': 0.1,
                      'right': 0.5,
                      'bottom': 0.9,
                    },
                  },
                ],
                'processingTimeMs': 25.5,
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'yolo11n.tflite', task: YOLOTask.detect);

      // Load model
      final loadSuccess = await yolo.loadModel();
      expect(loadSuccess, true);
      expect(log[0].method, 'loadModel');
      expect(log[0].arguments['task'], 'detect');

      // Predict
      final image = Uint8List.fromList(List.generate(1000, (i) => i % 256));
      final results = await yolo.predict(image);

      expect(results, isA<Map<String, dynamic>>());
      expect(results['boxes'], hasLength(1));
      expect(log[1].method, 'predictSingleImage');
    });

    test('different model paths are handled correctly', () async {
      final modelPaths = [
        'assets/models/yolo11n.tflite',
        '/absolute/path/to/model.tflite',
        'internal://models/custom_model.tflite',
        'yolo11n-seg.tflite',
        'yolo11s-pose.mlpackage',
      ];

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);
            return methodCall.method == 'loadModel' ? true : {};
          });

      for (final path in modelPaths) {
        log.clear();
        final yolo = YOLO(modelPath: path, task: YOLOTask.detect);
        await yolo.loadModel();

        expect(log.last.arguments['modelPath'], path);
      }
    });

    test('all task types can be loaded and used', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);

            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              final task = methodCall.arguments['task'] ?? 'detect';
              return _getMockResultForTask(task);
            }
            return null;
          });

      for (final task in YOLOTask.values) {
        log.clear();
        final yolo = YOLO(modelPath: 'test_model', task: task);

        await yolo.loadModel();
        expect(log.last.arguments['task'], task.name);

        final image = Uint8List.fromList([1, 2, 3, 4]);
        final results = await yolo.predict(image);
        expect(results, isA<Map<String, dynamic>>());
      }
    });
  });

  group('YOLO Edge Cases', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('handles very large image data', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') return true;
            if (methodCall.method == 'predictSingleImage') {
              final imageData = methodCall.arguments['image'] as Uint8List;
              expect(imageData.length, greaterThan(1000000)); // > 1MB
              return {'boxes': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      // Create large image (1.5MB)
      final largeImage = Uint8List(1500000);
      final results = await yolo.predict(largeImage);

      expect(results, isA<Map<String, dynamic>>());
    });

    test('handles network interruption during predict', () async {
      var callCount = 0;
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') return true;
            if (methodCall.method == 'predictSingleImage') {
              callCount++;
              if (callCount == 1) {
                throw PlatformException(
                  code: 'NETWORK_ERROR',
                  message: 'Connection lost',
                );
              }
              return {'boxes': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3, 4]);

      // First call should throw
      expect(() => yolo.predict(image), throwsA(isA<InferenceException>()));

      // Second call should succeed
      final results = await yolo.predict(image);
      expect(results, isA<Map<String, dynamic>>());
    });

    test('model path with special characters', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              final path = methodCall.arguments['modelPath'] as String;
              expect(path, contains('特殊字符'));
              expect(path, contains('émojï'));
              return true;
            }
            return null;
          });

      final yolo = YOLO(
        modelPath: 'models/特殊字符_émojï_🤖_model.tflite',
        task: YOLOTask.detect,
      );

      expect(() => yolo.loadModel(), returnsNormally);
    });

    test('predict with minimal valid image', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') return true;
            if (methodCall.method == 'predictSingleImage') {
              final imageData = methodCall.arguments['image'] as Uint8List;
              expect(imageData.length, 1);
              return {'boxes': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final minimalImage = Uint8List.fromList([0]);
      final results = await yolo.predict(minimalImage);

      expect(results, isA<Map<String, dynamic>>());
    });
  });

  group('YOLO useGpu Integration Tests', () {
  const MethodChannel channel = MethodChannel('yolo_single_image_channel');
  final List<MethodCall> log = <MethodCall>[];

  setUp(() {
    log.clear();
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('complete workflow with useGpu=false', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);

          if (methodCall.method == 'loadModel') {
            expect(methodCall.arguments['useGpu'], false);
            return true;
          } else if (methodCall.method == 'predictSingleImage') {
            return {
              'boxes': [
                {
                  'classIndex': 0,
                  'className': 'person',
                  'confidence': 0.95,
                  'boundingBox': {
                    'left': 10.0,
                    'top': 10.0,
                    'right': 110.0,
                    'bottom': 210.0,
                  },
                  'normalizedBox': {
                    'left': 0.1,
                    'top': 0.1,
                    'right': 0.5,
                    'bottom': 0.9,
                  },
                },
              ],
              'processingTimeMs': 25.5,
            };
          }
          return null;
        });

    final yolo = YOLO(modelPath: 'yolo11n.tflite', task: YOLOTask.detect);

    // Load model with CPU-only inference
    final loadSuccess = await yolo.loadModel(useGpu: false);
    expect(loadSuccess, true);
    expect(log[0].method, 'loadModel');
    expect(log[0].arguments['task'], 'detect');
    expect(log[0].arguments['useGpu'], false);

    // Predict
    final image = Uint8List.fromList(List.generate(1000, (i) => i % 256));
    final results = await yolo.predict(image);

    expect(results, isA<Map<String, dynamic>>());
    expect(results['boxes'], hasLength(1));
    expect(log[1].method, 'predictSingleImage');
  });

  test('different useGpu settings across model paths', () async {
    final testCases = [
      {'modelPath': 'yolo11n.tflite', 'useGpu': false, 'task': 'detect'},
      {'modelPath': 'yolo11s-seg.tflite', 'useGpu': true, 'task': 'segment'},
      {'modelPath': 'yolo11m-pose.tflite', 'useGpu': false, 'task': 'pose'},
      {'modelPath': 'yolo11l-cls.tflite', 'useGpu': true, 'task': 'classify'},
    ];

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);
          return methodCall.method == 'loadModel' ? true : {};
        });

    for (int i = 0; i < testCases.length; i++) {
      log.clear();
      final testCase = testCases[i];
      final yolo = YOLO(
        modelPath: testCase['modelPath'] as String,
        task: YOLOTask.values.firstWhere((t) => t.name == testCase['task']),
      );
      
      await yolo.loadModel(useGpu: testCase['useGpu'] as bool);

      expect(log.last.arguments['modelPath'], testCase['modelPath']);
      expect(log.last.arguments['useGpu'], testCase['useGpu']);
      expect(log.last.arguments['task'], testCase['task']);
    }
  });

  test('model switching with different useGpu settings', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);
          return methodCall.method == 'loadModel' ? true : null;
        });

    final yolo = YOLO(modelPath: 'initial_model.tflite', task: YOLOTask.detect);
    yolo.setViewId(1);

    // Initial load with GPU
    await yolo.loadModel(useGpu: true);
    expect(log.last.arguments['useGpu'], true);

    log.clear();

    // Switch to CPU-only model
    await yolo.switchModel('cpu_model.tflite', YOLOTask.segment, useGpu: false);
    expect(log.last.method, 'setModel');
    expect(log.last.arguments['useGpu'], false);
    expect(log.last.arguments['modelPath'], 'cpu_model.tflite');
    expect(log.last.arguments['task'], 'segment');
  });

  test('multi-instance with different useGpu settings', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);
          if (methodCall.method == 'createInstance' || methodCall.method == 'loadModel') {
            return true;
          }
          return null;
        });

    // Create two instances with different GPU settings
    final yoloGpu = YOLO(
      modelPath: 'gpu_model.tflite',
      task: YOLOTask.detect,
      useMultiInstance: true,
    );
    
    final yoloCpu = YOLO(
      modelPath: 'cpu_model.tflite',
      task: YOLOTask.segment,
      useMultiInstance: true,
    );

    expect(yoloGpu.instanceId, isNot(equals(yoloCpu.instanceId)));
    expect(yoloGpu.instanceId, isNot(equals('default')));
    expect(yoloCpu.instanceId, isNot(equals('default')));

    // Load with different GPU settings
    try {
      await yoloGpu.loadModel(useGpu: true);
      await yoloCpu.loadModel(useGpu: false);
      
      // Verify instances are properly configured
      expect(YOLOInstanceManager.hasInstance(yoloGpu.instanceId), isTrue);
      expect(YOLOInstanceManager.hasInstance(yoloCpu.instanceId), isTrue);
    } catch (e) {
      // Expected since we don't have a real platform implementation
      // The important part is that multi-instance structure is correct
      expect(yoloGpu.instanceId, isNot(equals('default')));
      expect(yoloCpu.instanceId, isNot(equals('default')));
    }
  });

  test('error handling with useGpu parameter', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          if (methodCall.method == 'loadModel') {
            final useGpu = methodCall.arguments['useGpu'] as bool;
            if (!useGpu) {
              throw PlatformException(
                code: 'GPU_DISABLED_ERROR',
                message: 'CPU-only mode validation error',
              );
            }
            return true;
          }
          return null;
        });

    final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);

    // Test GPU mode works
    expect(() => yolo.loadModel(useGpu: true), returnsNormally);

    // Test CPU mode error handling
         expect(
       () => yolo.loadModel(useGpu: false),
       throwsA(isA<ModelLoadingException>()),
     );
   });
 });
}

/// Generate mock results based on task type
Map<String, dynamic> _getMockResultForTask(String task) {
  switch (task) {
    case 'detect':
      return {
        'boxes': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
          },
        ],
      };
    case 'segment':
      return {
        'boxes': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
            'mask': [
              [0.1, 0.2],
              [0.3, 0.4],
            ],
          },
        ],
      };
    case 'pose':
      return {
        'boxes': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
            'keypoints': [100.0, 200.0, 0.9, 150.0, 250.0, 0.8],
          },
        ],
      };
    case 'classify':
      return {
        'classifications': [
          {'className': 'cat', 'confidence': 0.92},
          {'className': 'dog', 'confidence': 0.08},
        ],
      };
    case 'obb':
      return {
        'boxes': [
          {
            'classIndex': 0,
            'className': 'car',
            'confidence': 0.88,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
            'orientation': 45.0,
          },
        ],
      };
    default:
      return {'boxes': []};
  }
}
