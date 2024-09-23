import math
import random
import sys
import pygame
import numpy as np
import copy
import matplotlib.pyplot as plt

pygame.init()


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


TRACK = scale_image(pygame.image.load("imgs/track.jpg"), 1.1)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track_white.png"), 1.1)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs/finish.png")

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.7)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.7)
CAR_W, CAR_H = RED_CAR.get_width(), RED_CAR.get_height()

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

BORDER_COLOR = (255, 255, 255, 255)

best_fitnesses = []
avg_fitnesses = []
mutation_prob_wb = []
mutation_prob_layer = []

FPS = 60

max_time = FPS * 10
tick = 0
time = 0

DRAW_RADARS = False

class Car:
    def __init__(self):
        self.sprite = RED_CAR
        self.max_speed = 16
        self.min_speed = 8
        self.speed = 0
        self.rotation_vel = 8
        self.acceleration = 0.2
        self.angle = -90
        self.x, self.y = 320, 228
        self.width, self.height = CAR_W, CAR_H
        self.center = [self.x + self.width / 2, self.y + self.height / 2]
        self.radars = []
        self.distance = 0
        self.alive = True
        self.time = 0

    def draw_line(self, screen, angle, game_map):
        length = 1
        x = self.center[0] + math.cos((math.radians(self.angle + angle))) * length
        y = self.center[1] - math.sin((math.radians(self.angle + angle))) * length

        while not game_map.get_at((int(x), int(y))) == BORDER_COLOR and length < 200:
            length += 1
            x = self.center[0] + math.cos((math.radians(self.angle + angle))) * length
            y = self.center[1] - math.sin((math.radians(self.angle + angle))) * length
        if DRAW_RADARS:
            pygame.draw.line(screen, (0, 255, 0), self.center, (x, y), 1)
            pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)
        self.radars.append(length / 200)

    def draw(self, screen, game_map):
        rotated_image = pygame.transform.rotate(self.sprite, self.angle)
        new_rect = rotated_image.get_rect(center=self.sprite.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotated_image, new_rect.topleft)
        self.draw_line(screen, 90, game_map)
        self.draw_line(screen, 45, game_map)
        self.draw_line(screen, 0, game_map)
        self.draw_line(screen, -45, game_map)
        self.draw_line(screen, -90, game_map)

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        if right:
            self.angle -= self.rotation_vel

    def collide(self, mask):
        rotated_image = pygame.transform.rotate(self.sprite, self.angle)
        car_mask = pygame.mask.from_surface(rotated_image)
        new_rect = rotated_image.get_rect(center=self.sprite.get_rect(topleft=(self.x, self.y)).center)
        poi = mask.overlap(car_mask, new_rect.topleft)
        return poi

    def update(self):
        if self.speed == 0:
            self.speed = self.min_speed
        if self.speed < self.min_speed:
            self.speed = self.min_speed
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        radians = math.radians(self.angle)
        horizontal = math.cos(radians) * self.speed
        vertical = math.sin(radians) * self.speed

        self.distance += np.sqrt(horizontal ** 2 + vertical ** 2)

        if self.alive:
            self.y -= vertical
            self.x += horizontal

        self.center = [self.x + self.width / 2, self.y + self.height / 2]
        self.radars = []

        self.time += 1


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.num_neurons = n_neurons
        self.num_inputs = n_inputs

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def add_neuron(self):
        new_neuron = 0.1 * np.random.randn(self.num_inputs, 1)
        self.weights = np.append(self.weights, new_neuron, axis=1)
        self.biases = np.array([np.append(self.biases[0], [0], axis=0)])
        self.num_neurons += 1

    def add_weights(self):
        new_weights = 0.1 * np.random.randn(1, self.num_neurons)
        self.weights = np.append(self.weights, new_weights, axis=0)
        self.num_inputs += 1

    def remove_neuron(self):
        check_shape = (self.num_inputs, 1)
        if self.weights.shape != check_shape:
            self.weights = np.delete(self.weights, 0, axis=1)
            self.biases = np.delete(self.biases, 0, axis=1)
            self.num_neurons -= 1

    def remove_weights(self):
        check_shape = (1, self.num_neurons)
        if self.weights.shape != check_shape:
            self.weights = np.delete(self.weights, 0, axis=0)
            self.num_inputs -= 1

    def __str__(self):
        return "Weights:\n" + str(self.weights) + "\nBiases:" + str(self.biases)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs


class NeuralNetwork:
    def __init__(self, number_per_layer, activation_function, softmax):
        self.npl = number_per_layer
        self.layers = []
        self.activation_function = activation_function
        self.softmax = softmax
        self.fitness = 0
        for i in range(len(number_per_layer) - 1):
            self.layers.append(Layer_Dense(number_per_layer[i], number_per_layer[i + 1]))

    def forward(self, inputs):
        current_inputs = inputs
        current_outputs = None
        for layer in self.layers:
            layer.forward(current_inputs)
            current_outputs = layer.output
            self.activation_function.forward(layer.output)
            current_inputs = self.activation_function.output
        if self.softmax:
            exp_values = np.exp(current_outputs - np.max(current_outputs, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probs
        else:
            self.output = current_outputs

    def __str__(self):
        out = ""
        for i, lay in enumerate(self.layers):
            out += "Layer " + str(i) + ": \n" + str(lay) + "\n"
        return "-----Neural Network-----\nFitness: " + str(self.fitness) + "\n" + out


class GeneticAlgorithm:
    def __init__(self, population_size,
                 elitism_size,
                 elitism,
                 selection_size,
                 mutation_prob_wb_func,
                 mutation_prob_layer_func,
                 nn_structure,
                 activation_func,
                 goal_func):

        self.population_size = population_size
        self.elitism = elitism
        self.elitism_size = elitism_size + elitism_size % 2
        self.selection_size = selection_size
        self.mutation_prob_wb_func = mutation_prob_wb_func
        self.mutation_prob_layer_func = mutation_prob_layer_func
        self.nn_structure = nn_structure
        self.activation_func = activation_func
        self.goal_func = goal_func
        self.alive_individuals = population_size
        self.population = [NeuralNetwork(self.nn_structure, self.activation_func, True) for _ in range(population_size)]
        self.epoch = 0
        self.epoch_time = 0
        self.local_max_counter = 0
        self.divider = 1
        self.max_fitness = 0

    def new_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = copy.deepcopy(self.population)
        self.alive_individuals = self.population_size
        #
        #
        #
        #
        #
        #print(new_population[0])
        #
        #
        #
        #
        #
        #
        if self.max_fitness == new_population[0].fitness and self.max_fitness < 4000:
            self.local_max_counter += 1
        elif new_population[0].fitness > self.max_fitness:
            self.max_fitness = new_population[0].fitness
            self.local_max_counter = 0

        if self.local_max_counter >= 10:
            self.divider = 1

        best_fitnesses.append(new_population[0].fitness)
        tmp = 0
        for indv in new_population:
            tmp += indv.fitness
        tmp = tmp / len(new_population)
        avg_fitnesses.append(tmp)

        if self.elitism:
            for i in range(self.elitism_size, self.population_size, 2):
                parent1, parent2 = self.selection(self.population, self.selection_size)
                new_population[i], new_population[i + 1] = self.crossover_layers(parent1, parent2)
                new_population[i].fitness = 0
                new_population[i + 1].fitness = 0
                self.mutation_layer(new_population[i], self.mutation_prob_layer_func(self.epoch_time))
                self.mutation_layer(new_population[i + 1], self.mutation_prob_layer_func(self.epoch_time))
                self.mutation_wb(new_population[i], self.mutation_prob_wb_func(self.divider))
                self.mutation_wb(new_population[i + 1], self.mutation_prob_wb_func(self.divider))
            mutation_prob_wb.append(self.mutation_prob_wb_func(self.divider))
            mutation_prob_layer.append(self.mutation_prob_layer_func(self.epoch_time))
            self.population = copy.deepcopy(new_population)
        else:
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.selection(self.population, 20)
                new_population[i], new_population[i + 1] = self.crossover_layers(parent1, parent2)
                new_population[i].fitness = 0
                new_population[i + 1].fitness = 0
                self.mutation_layer(new_population[i], self.mutation_prob_layer_func(self.epoch_time))
                self.mutation_layer(new_population[i + 1], self.mutation_prob_layer_func(self.epoch_time))
                self.mutation_wb(new_population[i], self.mutation_prob_wb_func(self.divider))
                self.mutation_wb(new_population[i + 1], self.mutation_prob_wb_func(self.divider))
            mutation_prob_wb.append(self.mutation_prob_wb_func(self.divider))
            mutation_prob_layer.append(self.mutation_prob_layer_func(self.epoch_time))
            self.population = copy.deepcopy(new_population)
        self.epoch += 1
        self.divider += 1
        self.epoch_time = 0

    def calc_fitness(self, individual, tmp_car):
        individual.fitness = self.goal_func(tmp_car)

    def mutation_wb(self, indv, mutation_prob):
        for i in range(len(indv.layers)):
            if random.random() < 0.8:
                for j in range(len(indv.layers[i].weights)):
                    for k in range(len(indv.layers[i].weights[j])):
                        if random.random() < mutation_prob:
                            if random.random() < 0.5:
                                indv.layers[i].weights[j][k] += (random.random() - 0.5) * 0.1
                            else:
                                indv.layers[i].weights[j][k] -= (random.random() - 0.5) * 0.1

            else:
                for j in range(len(indv.layers[i].biases[0])):
                    if random.random() < mutation_prob * 0.5:
                        if random.random() < 0.5:
                            indv.layers[i].biases[0][j] += (random.random() - 0.5) * 0.1
                        else:
                            indv.layers[i].biases[0][j] -= (random.random() - 0.5) * 0.1

    def mutation_layer(self, indv, mutation_prob):
        if random.random() < mutation_prob:
            chosen = random.randrange(len(indv.layers) - 1)
            if random.random() < 0.5:
                indv.layers[chosen].remove_neuron()
                indv.layers[chosen + 1].remove_weights()
            else:
                indv.layers[chosen].add_neuron()
                indv.layers[chosen + 1].add_weights()

    def crossover_layers(self, par1, par2):
        chosen = random.randrange(0, len(par1.layers) - 1)
        delta_inputs = abs(par1.layers[chosen].num_inputs - par2.layers[chosen].num_inputs)
        if delta_inputs == 0:
            tmp_layer1 = copy.deepcopy(par1.layers[chosen])
            tmp_layer2 = copy.deepcopy(par2.layers[chosen])

            new_cld1 = copy.deepcopy(par2)
            for i in range(chosen):
                new_cld1.layers[i] = copy.deepcopy(par1.layers[i])
            new_cld1.layers[chosen] = copy.deepcopy(tmp_layer2)

            new_cld2 = copy.deepcopy(par1)
            for i in range(chosen):
                new_cld2.layers[i] = copy.deepcopy(par2.layers[i])
            new_cld2.layers[chosen] = copy.deepcopy(tmp_layer1)

            return new_cld1, new_cld2

        else:
            tmp_layer1 = copy.deepcopy(par1.layers[chosen])
            tmp_layer2 = copy.deepcopy(par2.layers[chosen])

            if tmp_layer1.num_inputs > tmp_layer2.num_inputs:
                for i in range(delta_inputs):
                    tmp_layer1.weights = np.delete(tmp_layer1.weights, tmp_layer1.num_inputs - 1, 0)
                    tmp_layer1.num_inputs -= 1
                    tmp_layer2.weights = np.vstack([tmp_layer2.weights, np.zeros((1, tmp_layer2.num_neurons))])
                    tmp_layer2.num_inputs += 1

                new_cld1 = copy.deepcopy(par2)
                for i in range(chosen):
                    new_cld1.layers[i] = copy.deepcopy(par1.layers[i])
                new_cld1.layers[chosen] = copy.deepcopy(tmp_layer2)

                new_cld2 = copy.deepcopy(par1)
                for i in range(chosen):
                    new_cld2.layers[i] = copy.deepcopy(par2.layers[i])
                new_cld2.layers[chosen] = copy.deepcopy(tmp_layer1)

                return new_cld1, new_cld2

            else:
                for i in range(delta_inputs):
                    tmp_layer2.weights = np.delete(tmp_layer2.weights, tmp_layer2.num_inputs - 1, 0)
                    tmp_layer2.num_inputs -= 1
                    tmp_layer1.weights = np.vstack([tmp_layer1.weights, np.zeros((1, tmp_layer1.num_neurons))])
                    tmp_layer1.num_inputs += 1

                new_cld1 = copy.deepcopy(par2)
                for i in range(chosen):
                    new_cld1.layers[i] = copy.deepcopy(par1.layers[i])
                new_cld1.layers[chosen] = copy.deepcopy(tmp_layer2)

                new_cld2 = copy.deepcopy(par1)
                for i in range(chosen):
                    new_cld2.layers[i] = copy.deepcopy(par2.layers[i])
                new_cld2.layers[chosen] = copy.deepcopy(tmp_layer1)

                return new_cld1, new_cld2

    def selection(self, population, indv_num):
        chosen = random.sample(population, indv_num)
        chosen.sort(key=lambda x: x.fitness, reverse=True)
        return chosen[0], chosen[1]


def wb_func(divider):
    return 1/max(1, divider)


def layer_func(epoch_time):
    return (max_time-epoch_time)/max_time


def fitness_func(tmp_car):
    return tmp_car.distance*(np.sqrt(tmp_car.time/max_time))


SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Bot")

clock = pygame.time.Clock()

base_font = pygame.font.Font(None, 32)

population_size = 20
elitism_size = max(2, int(population_size/10))
selection_size = max(2, int(population_size/20))
nn_structure = [5, 1, 1, 5]

gen_alg = GeneticAlgorithm(population_size,
                           elitism_size,
                           True,
                           selection_size,
                           wb_func,
                           layer_func,
                           nn_structure,
                           Activation_ReLU(),
                           fitness_func)

cars = [Car() for _ in range(population_size)]

while True:
    tick += 1
    time += 1
    gen_alg.epoch_time += 1
    clock.tick(FPS)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                DRAW_RADARS = not DRAW_RADARS
            if event.key == pygame.K_LEFT:
                FPS = max(30, int(FPS / 2))
            if event.key == pygame.K_RIGHT:
                FPS = min(240, int(FPS * 2))
        if event.type == pygame.QUIT:
            pygame.quit()
            plt.figure()
            plt.xlabel('Epoch')
            plt.plot([i for i in range(len(best_fitnesses))], best_fitnesses, label='Best Fitness')
            plt.plot([i for i in range(len(best_fitnesses))], avg_fitnesses, label='Average Fitness')
            plt.legend()
            plt.figure()
            plt.xlabel('Epoch')
            plt.plot([i for i in range(len(best_fitnesses))], mutation_prob_wb, label='Mutation Prob WB')
            plt.plot([i for i in range(len(best_fitnesses))], mutation_prob_layer, label='Mutation Prob Layer')
            plt.legend()
            plt.show()
            sys.exit(0)

    SCREEN.blit(TRACK, (0, 0))
    # SCREEN.blit(TRACK_BORDER, (0, 0))
    inputs = []
    car_cmds = []
    for i, car in enumerate(cars):
        car.draw(SCREEN, TRACK)
        if car.alive:
            gen_alg.population[i].forward(car.radars)
            # print(indv.output)
            car_cmds.append(np.argmax(gen_alg.population[i].output))
        else:
            car_cmds.append(-1)
    cars[0].draw(SCREEN, TRACK)

    for i, car_cmd in enumerate(car_cmds):
        if car_cmd == -1:
            continue
        elif car_cmd == 0:
            if cars[i].speed >= 1.0:
                cars[i].rotate(left=True)
        elif car_cmd == 1:
            cars[i].speed -= cars[i].acceleration
        elif car_cmd == 2:
            cars[i].speed += cars[i].acceleration
        elif car_cmd == 3:
            cars[i].speed -= cars[i].acceleration * 2
        else:
            if cars[i].speed >= 1.0:
                cars[i].rotate(right=True)

        if cars[i].alive:
            if cars[i].collide(TRACK_BORDER_MASK) is not None or cars[i].time >= max_time:
                cars[i].alive = False
                cars[i].update()
                gen_alg.calc_fitness(gen_alg.population[i], cars[i])
                gen_alg.alive_individuals -= 1
            else:
                cars[i].update()

    if gen_alg.alive_individuals == 0:
        gen_alg.new_generation()
        for i in range(len(cars)):
            cars[i] = Car()
        cars[0].sprite = GREEN_CAR

    text = str(FPS)
    text_surface = base_font.render(text, True, (0, 0, 0))
    SCREEN.blit(text_surface, (5, HEIGHT - 25))
