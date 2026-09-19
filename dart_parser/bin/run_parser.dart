import 'dart:convert';
import 'dart:io';

import '../lib/message_parser_pipeline.dart';

void main(List<String> arguments) async {
  if (arguments.length < 2) {
    print('Usage: dart run scripts/run_dart_parser.dart <input_jsonl> <output_jsonl>');
    exit(1);
  }

  final inputFile = File(arguments[0]);
  final outputFile = File(arguments[1]);

  if (!inputFile.existsSync()) {
    print('Error: Input file not found.');
    exit(1);
  }

  final lines = await inputFile.readAsLines();
  final outSink = outputFile.openWrite();
  
  int successCount = 0;
  int errorCount = 0;

  for (var line in lines) {
    if (line.trim().isEmpty) continue;
    
    try {
      final data = jsonDecode(line) as Map<String, dynamic>;
      final smsBody = data['body'] as String;
      final sender = data['sender'] as String?;
      
      final result = MessageParserPipeline.instance.parse(smsBody, sender: sender);
      
      final outMap = {
        'id': data['id'],
        'raw_body': smsBody,
        'sender': sender,
        'is_financial': result.isSuccess,
      };
      
      if (result.isSuccess && result.payment != null) {
        final p = result.payment!;
        outMap['parsed'] = {
          'amount': p.amount,
          'merchant': p.description,
          'type': p.type.name, // 'debit' or 'credit'
          'payment_mode': p.paymentMode.name,
          'account_ref': p.accountReference,
          'category': p.category,
        };
        successCount++;
      } else {
        outMap['error'] = result.errorMessage;
        errorCount++;
      }
      
      outSink.writeln(jsonEncode(outMap));
    } catch (e) {
      print('Error parsing line: $e');
    }
  }

  await outSink.close();
  print('Finished parsing.');
  print('Successfully parsed $successCount financial transactions.');
  print('Ignored/Failed: $errorCount messages.');
}
