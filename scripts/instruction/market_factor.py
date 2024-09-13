"""
Market factor fine-tuning dataset.
"""

market_factor_fine_tuning_dataset = {
    "attn": {
        "strategy_name": "attention",
        "rationale": "This strategy is based on the factor attention. \
The attention is positively correlated with future crypto market cumulative returns. \
Here's a detailed explanation of the rationale:\n"
        + "1. Google Search as a Measure of Attention\n"
        + "First, Internet users commonly use a search engine to collect \
information, and Google continues to be the favorite. The search volume \
reported by Google is thus likely to be representative of the internet \
search behavior of the general population. Second, and more critically,\
search is a revealed attention measure: if you search for a stock in \
Google, you are undoubtedly paying attention to it. Therefore, aggregate \
search frequency in Google is a direct and unambiguous measure of attention."
        + "2. Bitcoin as a Representative Cryptocurrency\n"
        + "attention use Google searches for the Bitcoin to proxy for \
investor attention of the cryptocurrency market because Bitcoin is by far \
the largest and most visible cryptocurrency available.\n"
        + "3. Predictability\n"
        + "Buying allows individuals to choose from a large set of \
alternatives while selling does not. For retail traders who rarely \
short, selling an asset requires individuals to have already owned \
the asset. Therefore, attention shocks lead to net buying by retail \
traders. Because retail traders are uninformed on average, this \
should lead to temporarily higher returns.",
    },
    "net": {
        "strategy_name": "network",
        "rationale": "This strategy is based on network factors. \
Network factors are positively correlated with future crypto market cumulative returns. \
Here's a detailed explanation of the rationale:\n"
        + "1. Bitcoin network as a Representative Network\n"
        + "Because Bitcoin is by far the largest and well-known \
cryptocurrency available, we use Bitcoin network.\n"
        + "2. Predictability\n"
        + "The network effect of user adoption can potentially play \
a central role in the valuation of cryptocurrencies. Because users' adoption \
of cryptocurrencies generates positive network externality, cryptocurrency \
prices respond to user adoptions. Hence, variations in user adoptions of the \
cryptocurrency network could contribute to movements in cryptocurrency \
prices.",
    },
}

# market_factor_fine_tuning_dataset = [
#     # Liu et al. (2021) Risk and Return of Cryptocurrency
#     # Attention
#     {
#         "strategy": "attention",
#         "description": "Calculate the Google search data for Bitcoin minus \
# its average of the previous four weeks. Standardize the data with a mean of \
# zero and a standard deviation of one. Grouping them into terciles: low, \
# medium, and high.",
#         "monotonicity": "increasing",
#         "rationale": "The allocation strategy is based on the factor attention. \
# The attention is positively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Google Search as a Measure of Attention:\n"
#         + "First, Internet users commonly use a search engine to collect \
# information, and Google continues to be the favorite. The search volume \
# reported by Google is thus likely to be representative of the internet \
# search behavior of the general population. Second, and more critically,\
# search is a revealed attention measure: if you search for a stock in \
# Google, you are undoubtedly paying attention to it. Therefore, aggregate \
# search frequency in Google is a direct and unambiguous measure of attention."
#         + "Bitcoin as a Representative Cryptocurrency:\n"
#         + "attention use Google searches for the Bitcoin to proxy for \
# investor attention of the cryptocurrency market because Bitcoin is by far \
# the largest and most visible cryptocurrency available.\n"
#         + "Predictability:\n"
#         + "Buying allows individuals to choose from a large set of \
# alternatives while selling does not. For retail traders who rarely \
# short, selling an asset requires individuals to have already owned \
# the asset. Therefore, attention shocks lead to net buying by retail \
# traders. Because retail traders are uninformed on average, this \
# should lead to temporarily higher returns.",
#     },
#     # Wallet user growth
#     {
#         "strategy": "wallet_user_growth",
#         "description": "Calculate the growth of wallet user in Bitcoin network.",
#         "monotonicity": "increasing",
#         "rationale": "The allocation strategy is based on the factor wallet_user_growth. \
# The wallet_user_growth is positively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Bitcoin network as a Representative Network:\n"
#         + "Because Bitcoin is by far the largest and well-known \
# cryptocurrency available, we use Bitcoin network.\n"
#         + "Predictability:\n"
#         + "The network effect of user adoption can potentially play \
# a central role in the valuation of cryptocurrencies. Because users' adoption \
# of cryptocurrencies generates positive network externality, cryptocurrenc \
# prices respond to user adoptions. Hence, variations in user adoptions of the \
# cryptocurrency network could contribute to movements in cryptocurrency \
# prices.",
#     },
#     # Active address growth
#     {
#         "strategy": "active_address_growth",
#         "description": "Calculate the growth of active address in Bitcoin network.",
#         "monotonicity": "increasing",
#         "rationale": "The allocation strategy is based on the factor wallet_user_growth. \
# The wallet_user_growth is positively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Bitcoin network as a Representative Network:\n"
#         + "Because Bitcoin is by far the largest and well-known \
# cryptocurrency available, we use Bitcoin network.\n"
#         + "Predictability:\n"
#         + "The network effect of user adoption can potentially play \
# a central role in the valuation of cryptocurrencies. Because users' adoption \
# of cryptocurrencies generates positive network externality, cryptocurrency \
# prices respond to user adoptions. Hence, variations in user adoptions of the \
# cryptocurrency network could contribute to movements in cryptocurrency \
# prices.",
#     },
#     # Payment count growth
#     {
#         "strategy": "payment_count_growth",
#         "description": "Calculate the growth of payments in Bitcoin network.",
#         "monotonicity": "increasing",
#         "rationale": "The allocation strategy is based on the factor wallet_user_growth. \
# The wallet_user_growth is positively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Bitcoin network as a Representative Network:\n"
#         + "Because Bitcoin is by far the largest and well-known \
# cryptocurrency available, we use Bitcoin network.\n"
#         + "Predictability:\n"
#         + "The network effect of user adoption can potentially play \
# a central role in the valuation of cryptocurrencies. Because users' adoption \
# of cryptocurrencies generates positive network externality, cryptocurrency \
# prices respond to user adoptions. Hence, variations in user adoptions of the \
# cryptocurrency network could contribute to movements in cryptocurrency \
# prices.",
#     },
# ]

# signal allocation mapping
SIGNAL_ALLOC_MAPPING = {
    "LOW": 30,
    "MIDDLE": 50,
    "HIGH": 70,
}
