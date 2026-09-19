# Dataset Specification V2

## Overview
This document outlines the annotation strategy and the adversarial examples used to train the Intent Classifier and NER models for the Indian Bank SMS parsing system. 

## Taxonomy
- **Intent**: `TRANSACTION`, `REFUND`, `REVERSAL`, `FAILED`, `PENDING`, `BALANCE`, `BILL_REMINDER`, `PROMOTIONAL`, `OTP_SECURITY`, `SCAM`, `INFORMATIONAL`
- **Direction**: `DEBIT`, `CREDIT`, `NONE`
- **NER**: `TRANSACTION_AMOUNT`, `BALANCE_AMOUNT`, `MERCHANT`, `MERCHANT_VPA`, `ACCOUNT`, `CARD`, `REFERENCE_ID`, `BANK_NAME`, `TRANSACTION_DATE`

## Adversarial Examples (Dataset C)
To prevent the model from simply memorizing keywords like "Rs. 500" or "transaction", we include hard negative examples for easily confused classes.

### 1. TRANSACTION vs BILL_REMINDER
- **TRANSACTION**: `Rs.500 debited from your account XX1234 towards Swiggy.` (Intent: TRANSACTION, Dir: DEBIT)
- **BILL_REMINDER**: `Payment of Rs.500 for your credit card XX1234 is due tomorrow.` (Intent: BILL_REMINDER, Dir: NONE)
- **TRANSACTION**: `Paid Rs.25000 towards your credit card bill.` (Intent: TRANSACTION, Dir: DEBIT)
- **BILL_REMINDER**: `Please pay your due amount of Rs.25000 before 5th Oct to avoid late fees.` (Intent: BILL_REMINDER, Dir: NONE)

### 2. REFUND vs REVERSAL vs TRANSACTION
- **REFUND**: `Rs. 500 refund credited to your A/c XX1234 for your Amazon order.` (Intent: REFUND, Dir: CREDIT)
- **REVERSAL**: `Rs. 500 debited earlier has been reversed to your account.` (Intent: REVERSAL, Dir: CREDIT)
- **TRANSACTION**: `Rs. 500 credited to your account via IMPS from John.` (Intent: TRANSACTION, Dir: CREDIT)

### 3. FAILED vs PENDING vs TRANSACTION
- **FAILED**: `Rs.500 transaction failed due to insufficient funds.` (Intent: FAILED, Dir: DEBIT - attempted)
- **PENDING**: `Rs.500 transaction is pending and will be settled in 2 working days.` (Intent: PENDING, Dir: DEBIT)
- **TRANSACTION**: `Rs.500 debited. Available balance is Rs.1000.` (Intent: TRANSACTION, Dir: DEBIT)

### 4. TRANSACTION vs PROMOTIONAL
- **PROMOTIONAL**: `Get Rs.500 cashback on your next transaction with HDFC credit card.` (Intent: PROMOTIONAL, Dir: NONE)
- **TRANSACTION**: `Rs.500 cashback credited to your wallet.` (Intent: TRANSACTION, Dir: CREDIT)
- **PROMOTIONAL**: `Spend Rs.5000 on shopping and get 5X reward points.` (Intent: PROMOTIONAL, Dir: NONE)
- **TRANSACTION**: `Rs.5000 spent on shopping at Myntra using your card XX1234.` (Intent: TRANSACTION, Dir: DEBIT)

### 5. TRANSACTION vs BALANCE
- **TRANSACTION**: `Rs.500 debited. Your new balance is Rs.10500.` (Intent: TRANSACTION, Dir: DEBIT)
- **BALANCE**: `Your available account balance is Rs.10500 as on 10/10/2023.` (Intent: BALANCE, Dir: NONE)

### 6. NER Ambiguities (Amount vs Balance)
- `Rs. 500 debited. Available balance Rs 4,250.`
  - `TRANSACTION_AMOUNT`: 500
  - `BALANCE_AMOUNT`: 4250
- `Your payment of Rs. 200 was successful. Limit remaining: Rs. 1,00,000.`
  - `TRANSACTION_AMOUNT`: 200
  - `BALANCE_AMOUNT`: 100000

### 7. NER Ambiguities (Merchant vs VPA)
- `Paid Rs. 500 to swiggy@icici`
  - `MERCHANT`: swiggy
  - `MERCHANT_VPA`: swiggy@icici
- `UPI transaction to AMAZON INDIA for Rs. 500.`
  - `MERCHANT`: AMAZON INDIA
  - `MERCHANT_VPA`: null


## LLM Prompt Template for Auto-Labeling
```text
You are a highly accurate financial SMS annotator for Indian banking messages.
Given an SMS, you must classify its Intent, Direction, and extract specific Entities based on our strict taxonomy.

[INSERT TAXONOMY HERE]

Example 1:
SMS: "Rs.500 debited from your account XX1234 towards Swiggy. Bal: Rs.1000."
Output:
{
  "intent": "TRANSACTION",
  "direction": "DEBIT",
  "entities": {
    "TRANSACTION_AMOUNT": 500,
    "ACCOUNT": "XX1234",
    "MERCHANT": "Swiggy",
    "BALANCE_AMOUNT": 1000
  }
}

Example 2:
SMS: "Get Rs.500 cashback on your next transaction."
Output:
{
  "intent": "PROMOTIONAL",
  "direction": "NONE",
  "entities": {}
}

Analyze the following SMS:
SMS: "{sms_text}"
Output:
```
