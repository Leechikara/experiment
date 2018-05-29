# coding=utf-8


DATA_ROOT = '/home/wkwang/workstation/experiment/data'

# experiments for KB changes
ORIGINAL_ENTITY = 50
CHANGE_ORIGINAL_ENTITY = False
NEW_ADD_ENTITY = 0

# attribute definition of goods
GOODS_ATTRIBUTE_DEFINITION = {'price': {'dtype': 'int', 'min': 1000, 'max': 2000, 'prefer': 'low'},
                              'discountValue': {'dtype': 'float', 'range': [0.1 * x for x in range(6, 10)],
                                                'acceptNone': True, 'prefer': 'low'},
                              'discountURL': {'dtype': 'str', 'prefix': 'https://activity/', 'entity_id': None,
                                              'discountValue': None, 'postfix': '.html', 'correlate': 'discountValue'},
                              'weight': {'dtype': 'int', 'range': range(80, 100), 'expensive': 'low',
                                         'prefer': 'low'},
                              'os': {'dtype': 'str', 'range': ['Android', 'iOS']},
                              'color': {'dtype': 'list', 'range': ['blue', 'black', 'yellow', 'red']},
                              'thickness': {'dtype': 'float', 'range': [0.1 * x for x in range(30, 41)],
                                            'expensive': 'low', 'prefer': 'low'},
                              'size': {'dtype': 'float', 'range': [0.1 * x for x in range(90, 100)],
                                       'expensive': 'high', 'prefer': 'high'},
                              'material': {'dtype': 'str', 'range': ['塑料', '木头', '金属'], 'expensive': 'high',
                                           'prefer': 'high'},
                              'network': {'dtype': 'str', 'range': ['3g', '4g'], 'expensive': 'high',
                                          'prefer': 'high'},
                              'nfc': {'dtype': 'bool', 'range': [True, False], 'expensive': 'low',
                                      'prefer': 'low'},
                              'generation': {'dtype': 'str', 'range': [str(x) for x in range(1, 4)],
                                             'expensive': 'high', 'prefer': 'high'}}

# global attribute definitions
OTHER_ATTRIBUTE_DEFINITION = {
    'payment': {'dtype': 'constantList', 'value': ['微信', '支付宝', '信用卡']},
    'expressTime': {'dtype': 'constantInt', 'value': 3},
    'expressName': {'dtype': 'constantList', 'value': '顺丰'}}

# simulate noise in price
PRICE_NOISE = 300

# available attributes in different intents
QA_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'os', 'color', 'thickness', 'size', 'material',
                     'network', 'nfc', 'generation']
COMPARE_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'thickness', 'size', 'material',
                          'network', 'generation']
CONFIRM_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'os', 'color', 'thickness', 'size', 'material',
                          'network', 'nfc', 'generation']
PRE_SALES_ATTR = list(set(QA_PERMITTED_ATTR) | set(COMPARE_PERMITTED_ATTR) | set(CONFIRM_PERMITTED_ATTR))

# we simulate the size of Hypothesis Space
AVAILABLE_INTENT_1 = {'pre_sales': ['qa'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName']}

AVAILABLE_INTENT_2 = {'pre_sales': ['qa', 'confirm'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName']}

AVAILABLE_INTENT_3 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName']}

AVAILABLE_INTENT_4 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName'],
                      'after_sales': ['expressInfo', 'invoice', 'exchange']}

AVAILABLE_INTENT_5 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName'],
                      'after_sales': ['expressInfo', 'invoice', 'exchange', 'exchange_exchange', 'consult', 'refund',
                                      'consult_refund']}

AVAILABLE_INTENT_6 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'expressTime', 'expressName'],
                      'after_sales': ['expressInfo', 'invoice', 'exchange', 'exchange_exchange', 'consult', 'refund',
                                      'consult_refund'],
                      'sentiment': ['positive', 'negative']}

# intent distributions in different scenes
INTENT_P_DICT = {'pre_sales': {'qa': 1 / 3, 'confirm': 1 / 3, 'compare': 1 / 3},
                 'in_sales': {'discountURL': 1 / 4, 'payment': 1 / 4, 'expressTime': 1 / 4,
                              'expressName': 1 / 4},
                 'after_sales': {'consult': 1 / 7, 'refund': 1 / 7, 'exchange': 1 / 7, 'expressInfo': 1 / 7,
                                 'invoice': 1 / 7, 'consult_refund': 1 / 7, 'exchange_exchange': 1 / 7}}

# grammar distribution in different situations
# We can change this distributes to improve the quality of episode generation rules!
GRAMMAR_P_DICT = {'pre_sales': {'qa': {'init': {'complete': 1 / 30, 'lack_attribute': 1 / 3, 'lack_entity': 1 / 3},
                                       'horizontal_1': {'complete': 1 / 20, 'lack_attribute': 1 / 2},
                                       'vertical_1': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                       'diagonal_1': {'complete': 1 / 10}},
                                'confirm': {'init': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                            'horizontal_1': {'complete': 1 / 10},
                                            'vertical_1': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                            'diagonal_1': {'complete': 1 / 10}},
                                'compare': {'init': {'complete': 1 / 40, 'lack_entity': 1 / 4,
                                                     'lack_attribute': 1 / 4, 'lack_attribute_entity': 1 / 4},
                                            'horizontal_1': {'complete': 1 / 40, 'lack_entity': 1 / 4,
                                                             'lack_attribute': 1 / 4, 'lack_attribute_entity': 1 / 4},
                                            'diagonal_1': {'complete': 1 / 20},
                                            'horizontal_2': {'complete': 1 / 20, 'lack_attribute': 1 / 2},
                                            'diagonal_2': {'complete': 1 / 10}}},
                  'in_sales': {'discountURL': {'complete': 1 / 2, 'lack_entity': 1 / 2},
                               'payment': {'complete': 1},
                               'expressTime': {'complete': 1},
                               'expressName': {'complete': 1}}}
