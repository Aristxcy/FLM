import json
import os
import sys
import math

from python_choice_models.estimation.market_explore.ranked_list import MIPMarketExplorer
from python_choice_models.estimation.expectation_maximization.markov_chain import \
    MarkovChainExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.ranked_list import \
    RankedListExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.latent_class import \
    LatentClassExpectationMaximizationEstimator
from python_choice_models.estimation.maximum_likelihood.random_choice import RandomChoiceModelMaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from python_choice_models.estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator

from python_choice_models.settings import Settings

from python_choice_models.models import Model, MixedLogitModel, MultinomialLogitModel, ExponomialModel, \
    LatentClassModel, MarkovChainModel, MarkovChainRank2Model, NestedLogitModel, RandomChoiceModel, RankedListModel

from python_choice_models.transactions.base import Transaction

GLOBAL_TIME_LIMIT = 3600

NORMAL_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': None,
    'solver_total_time_limit': 1000.0,
}

RANKED_LIST_SETTINGS = {
    'linear_solver_partial_time_limit': 300,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 3600.0,
}

LATENT_CLASS_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 3600.0,
}


def read_json(file_name):
    with open(file_name, 'r') as f:
        data = json.loads(f.read())
    ground_truth = Model.from_data(data['ground_model'])
    transactions = Transaction.from_json(data['transactions']['in_sample_transactions'])
    test_set = Transaction.from_json(data['transactions']['out_of_sample_transactions'])
    return transactions, ground_truth, test_set


training_length = 6000
validation_length = 1500

for ii in range(10):
    input_1 = f"./ground_{validation_length}_{ii+1}_validation.json"
    input_2 = f"./ground_{training_length}_{ii+1}_train.json"

    ll_valid_set_benchmark = -math.inf
    ll_training_log_like = 0

    gammas = []
    mnls = []

    for j in range(1, 5):
        estimators = {
                'max': {
                    'name': 'Standard Maximum Likelihood',
                    'models': {
                        'lc': {
                            'estimator': MaximumLikelihoodEstimator(),
                            'model_class': lambda products: LatentClassModel.simple_deterministic(products, j+1),
                            'name': 'Latent Class',
                            'settings': RANKED_LIST_SETTINGS
                        }
                    }
                },
                'em': {
                    'name': 'Expectation Maximization',
                    'models': {
                        'lc': {
                            'estimator': LatentClassExpectationMaximizationEstimator(),
                            'model_class': lambda products: LatentClassModel.simple_deterministic(products, j+1),
                            'name': 'Latent Class',
                            'settings': RANKED_LIST_SETTINGS
                        }
                    }
                },
                'fw': {
                    'name': 'Frank Wolfe/Conditional Gradient',
                    'models': {
                        'lc': {
                            'estimator': LatentClassFrankWolfeEstimator(),
                            'model_class': lambda products: LatentClassModel.simple_deterministic(products, j+1),
                            'name': 'Latent Class',
                            'settings': NORMAL_SETTINGS
                        },
                    }
                }
        }

        transactions_validation, _, _ = read_json(input_1)
        transactions, ground_truth, _ = read_json(input_2)
        

        products = ground_truth.products
        estimation_method = 'max'
        model = 'lc'
        
        print(' * Under the ' + model + ' model:')

        print('- Amount of transactions: %s' % len(transactions))
        print('- Amount of products: %s' % len(products))
        model_info = estimators[estimation_method]['models'][model]

        print(' * Retrieving settings...')
        Settings.new(
            linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
            non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
            solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
        )
        print(' * Creating initial solution...')
        model = model_info['model_class'](products)

        print(' * Starting estimation...')
        if hasattr(model_info['estimator'], 'estimate_with_market_discovery'):
            result = model_info['estimator'].estimate_with_market_discovery(model, transactions)
        else:
            print('Here.')
            result = model_info['estimator'].estimate(model, transactions)

        log_training = result.log_likelihood_for(transactions) * len(transactions)
        log_validation = result.log_likelihood_for(transactions_validation) * len(transactions_validation)

        print(f'the loglikelihood for the training set: {log_training}')
        print(f'the loglikelihood for the validation set: {log_validation}')


        if log_validation > ll_valid_set_benchmark:
            ll_valid_set_benchmark = log_validation
            ll_training_log_like = log_training
            gammas = result.gammas
            mnls = result.mnl_models()


    with open(f'./ground_mmnl_{training_length}_{ii+1}_gamma.txt', 'w') as f:
        for item in gammas:
            f.write(f'{item}\t')
        f.write(f'{ll_training_log_like}\t')

    with open(f'./ground_mmnl_{training_length}_{ii+1}_mnl.txt', 'w') as f:
        num_types = len(gammas)
        for i in range(num_types):
            mnl = mnls[i].etas
            for item in mnl:
                f.write(f'{item}\t')
            f.write('\n')


