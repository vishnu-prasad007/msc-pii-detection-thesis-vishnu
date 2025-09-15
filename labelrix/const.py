PII_TYPES = [
    "Person Name",
    "Email Address",
    "Phone Number",
    # "Mobile Number",
    "Location",
    "Organization Name",
    "Date",
    # "Gender",
    # "ID Card Number",
    "Contract Number",
    "Invoice Number"
]

PII_EXTRA_RULES = {
    "Person Name": "Extract complete human names, including any preceding salutations (e.g., 'Dr. John Doe', 'Mr. Smith'). Do not include usernames, single words, or rewrite any names; always extract them exactly as they appear in the text",
    "Email Address": "Extract valid email addresses like 'example@mail.com'. Do not extract incomplete emails, or rewrite any emails; always extract them exactly as they appear in the text",
    "Phone Number": "Extract landline, mobile numbers or telephone numbers  including area codes if present. Ignore random numbers. Do not rewrite any Phone Number; always extract them exactly as they appear in the text",
    "Mobile Number": "Extract mobile phone numbers including country codes if present. Ignore random numbers. Do not rewrite any Mobile Number; always extract them exactly as they appear in the text",
    "Location": "Extract postal addresses. Do not rewrite any Location; always extract them exactly as they appear in the text",
    "Organization Name": "Extract company or institution names. Do not extract departments. Do not rewrite any Organization Name; always extract them exactly as they appear in the text",
    "Social Security Number": "Extract valid SSN format (e.g., '123-45-6789'). Ignore random numbers.",
    "Date": "Extract only complete calendar dates and times, such as 'MM/DD/YYYY HH:MM', 'YYYY-MM-DD HH:MM:SS', or 'Month Day, Year HH:MM AM/PM'. Do not extract age ranges (e.g., '18-24'), partial dates (e.g., 'May 2023'), or time ranges (e.g., '9-5'). Do not rewrite any Date; always extract them exactly as they appear in the text",
    "Gender": "Extract 'Male' or 'Female' only.",
    "ID Card Number": "Extract official ID numbers like passport or driver's license IDs. Ignore unrelated numbers.",
    "Contract Number": "Extract document contract identifiers. Ignore dates or monetary values. Do not rewrite any Contract Number; always extract them exactly as they appear in the text",
    "Invoice Number": "Extract invoice identifiers. Ignore amounts or dates."
}

# Allowed OCR document types (per-page classification)
ALLOWED_DOC_TYPES = {"letter", "report", "memo", "note", "email", "form"}

PII_EXTRA_RULES_VERIFICATION = {
    "Person Name": "Person Name are complete full names, it could be first name, last name, middle name, or any combination of these. It could be a single word or a combination of words. It could be a full name or a partial name. It could be a name with a prefix or a suffix. It could be a name with a title or a salutation. Person names include any preceding titles like 'Dr.', 'Mr.', 'Ms.' but exclude usernames or single unrelated words.",
    "Email Address": "A valid email address is a string of characters that follows the format: local-part@domain. Email addresses must contain '@' symbol and a valid domain structure. Incomplete or malformed email strings are not considered valid email addresses.",
    "Phone Number": "Phone numbers are landline, mobile numbers or telephone numbers that may include area codes, country codes, or formatting characters like dashes, parentheses, or spaces. Phone numbers may also include common prefixes or suffixes such as 'O:', 'M:', 'Tel:', 'Phone:', 'Office:', 'Mobile:', or similar indicators that commonly accompany phone numbers in documents. Random number sequences that don't represent actual phone numbers should be ignored.",
    "Mobile Number": "Mobile numbers are cellular phone numbers that may include country codes, area codes, or formatting characters. They represent valid mobile phone contact information and exclude random number sequences.",
    "Location": "Postal addresses are complete or partial physical addresses including street addresses, city names, states, postal codes, or any combination of location identifiers that represent a geographical location or mailing address.",
    "Organization Name": "Organization names are company, institution, business, or corporate entity names. They exclude internal departments or divisions and represent the main organizational identity.",
    "Social Security Number": "Social Security Numbers follow the standard SSN format with 9 digits, typically formatted as XXX-XX-XXXX. Random number sequences that don't match SSN patterns are excluded.",
    "Date": "Dates are complete calendar dates and times in various formats such as MM/DD/YYYY, YYYY-MM-DD, or written formats like 'Month Day, Year'. Age ranges, partial dates, or time ranges are excluded.",
    "Gender": "Gender information is limited to explicit mentions of 'Male' or 'Female' gender identifications.",
    "ID Card Number": "ID card numbers are official government-issued identification numbers such as passport numbers, driver's license numbers, or other official ID documents. Unrelated number sequences are excluded.",
    "Contract Number": "Contract numbers are unique identifiers assigned to legal agreements or contractual documents. They exclude dates, monetary values, or other unrelated numerical information.",
    "Invoice Number": "Invoice numbers are unique identifiers assigned to billing documents or payment requests. They exclude monetary amounts, dates, or other non-identifier information."
}


PROMPT_TEMPLATES = [
    # Variant 1
    "Instruction: Identify and extract all instances of {pii_type} from the provided text, appending (X|Y) tokens. Respond strictly in JSON between <json-start> and <json-end> tags in the forma:\n\n"
    "<json-start>\n"
    "{{\"values\": [\n"
    "  \"<text value> (X|Y)\",\n"
    "  \n"
    "]}}\n"
    "<json-end>\n\n"
    "Important rules:\n"
    "- NO explanations, comments, or extra text.\n"
    "- NO markdown formatting.\n"
    "- If no {pii_type} are found, output exactly:\n\n"
    "<json-start>\n"
    "{{\"values\": []}}\n"
    "<json-end>\n"
    "Additional guidance: {extra_rule}",

    # Variant 2
    "Instruction: Imagine you are a data privacy analyst scrutinizing the following text for sensitive information. Your task is to accurately locate and report all occurrences of {pii_type}, appending (X|Y) tokens. Respond strictly in JSON between <json-start> and <json-end> tags in the format:\n\n"
    "<json-start>\n"
    "{{\"values\": [\n"
    "  \"<text value> (X|Y)\",\n"
    "  \n"
    "]}}\n"
    "<json-end>\n\n"
    "Guidelines:\n"
    "- No extra text.\n"
    "- No explanations.\n"
    "- No markdown.\n"
    "- If no matches, output exactly:\n\n"
    "<json-start>\n"
    "{{\"values\": []}}\n"
    "<json-end>\n"
    "Additional guidance: {extra_rule}",

    # Variant 3
    "Instruction: Review the provided text to ensure no {pii_type} is missed. For every identified {pii_type} appending (X|Y) tokens. Return ONLY a JSON array as shown:\n\n"
    "<json-start>\n"
    "{{\"values\": [\n"
    "  \"<text value> (X|Y)\",\n"
    "  \n"
    "]}}\n"
    "<json-end>\n\n"
    "Do not include any other text, notes, or formatting. If none found, output exactly:\n\n"
    "<json-start>\n"
    "{{\"values\": []}}\n"
    "<json-end>\n"
    "Additional guidance: {extra_rule}"
]


VERIFICATION_PROMPT_TEMPLATES = [
    # Prompt 1: Definition-based Verification
    "Instruction: Given this document context, determine if each numbered {pii_type} span is actually {pii_type}. Respond strictly in JSON between <json-start> and <json-end> tags in the format:\n\n"
    "<json-start>\n"
    "{{\"Verifications\": [\n"
    "  \"1:Yes\",\n"
    "  \"2:No\",\n"
    "  \"3:Yes\"\n"
    "]}}\n"
    "<json-end>\n\n"
    "Document: {document}\n\n"
    "Spans to verify:\n"
    "{span_list}\n\n"
    "Important rules:\n"
    "- Answer only Yes or No for each numbered span.\n"
    "- NO explanations, comments, or extra text.\n"
    "- NO markdown formatting.\n"
    "- Format: \"number:Yes\" or \"number:No\n"
    "Additional guidance: {extra_rule}",

    # Prompt 2: Counter-evidence based Verification  
    "Instruction: Given this document context, identify reasons why each numbered {pii_type} span might NOT be {pii_type}. If no good reason exists, answer Yes. Respond strictly in JSON between <json-start> and <json-end> tags in the format:\n\n"
    "<json-start>\n"
    "{{\"Verifications\": [\n"
    "  \"1:Yes\",\n"
    "  \"2:No\",\n"
    "  \"3:Yes\"\n"
    "]}}\n"
    "<json-end>\n\n"
    "Document: {document}\n\n"
    "Spans to verify:\n"
    "{span_list}\n\n"
    "Important rules:\n"
    "- Look for counter-evidence first, then decide.\n"
    "- Answer only Yes or No for each numbered span.\n"
    "- NO explanations, comments, or extra text.\n"
    "- NO markdown formatting.\n"
    "- Format: \"number:Yes\" or \"number:No\n"
    "Additional guidance: {extra_rule}"
]