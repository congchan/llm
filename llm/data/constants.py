"""
Global constants.
"""

USER = "user"
ASSIS = "assistant"

PROCESS_DIR = "processed"

RP_SYSTEM = "Role-playing"
MTRP_SYSTEM = "Multi-roles-playing"
FUNC_SYSTEM = "Function-calling"

DEMO_SAMPLE = {
    'id': 'DEMO_00',
    'system': RP_SYSTEM,
    "background": "xxxyyyyzzz",
    "respond_style": "aaabbbccc",
    'conversations': [
        {'from': USER, 'value': "Hi"},
        {'from': ASSIS, 'value': "Hi"},
    ],
    'src': 'demo',
    'type': 'conversation'
}
