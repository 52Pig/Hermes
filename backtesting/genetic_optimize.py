import random
from deap import base, creator, tools, algorithms
import logging


def optimize_strategy_parameters(engine, strategy, param_space, n_generations=20, population_size=50):
    logging.info(f"开始优化策略 {strategy.name} 参数")

    def evaluate(individual):
        strategy.set_parameters(individual)
        engine.run_backtest()
        result = engine.strategy_results[strategy.name]
        return -result['portfolio_value'],  # 遗传算法最大化问题，这里取负值最小化

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, param_space[0], param_space[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(param_space))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    logging.info(f"最佳参数为: {best_individual}")
