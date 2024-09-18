import pygame
import time
import math
import random

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
GRASS_MASK = pygame.mask.from_surface(GRASS)
TRACK_MASK = pygame.mask.from_surface(TRACK)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs/finish.png")

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Bot")

GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)


FPS = 120
clock = pygame.time.Clock()

class Individual:
    def __init__(self):
        self.fitness = 0
        self.keys = []
        for _ in range(1920):
            rnd = random.random()
            if rnd <= 0.45:
                self.keys.append(pygame.K_w)
            elif rnd <= 0.70:
                self.keys.append(pygame.K_a)
            elif rnd <= 0.95:
                self.keys.append(pygame.K_d)
            else:
                self.keys.append(None)
    

def draw(win, images, bot_car):
    for img, pos in images:
        win.blit(img, pos)
    bot_car.draw(WIN)
    pygame.display.update()
images = [(GRASS, (0, 0)), (TRACK, (0, 0))]

def selection(population, size, forbidden):
    allowed = list(set(range(len(population))).difference({forbidden}))
    chosen = random.sample(allowed, size)
    max_fitness = -1
    best_index = -1
    for i in chosen:
        if population[i].fitness > max_fitness:
            max_fitness = population[i].fitness
            best_index = i
    return best_index

def crossover(parent1, parent2, child1, child2):
    pos = random.randrange(1, len(parent1.keys))
    
    child1.keys[:pos] = parent1.keys[:pos]
    child1.keys[pos:] = parent2.keys[pos:]
    
    child2.keys[:pos] = parent2.keys[:pos]
    child2.keys[pos:] = parent1.keys[pos:]

def mutation(child, mutation_prob = 0.01):
    for i in range(len(child.keys)):
        if random.random() < mutation_prob:
            rnd = random.random()
            if rnd <= 0.50:
                child.keys[i] = pygame.K_w
            elif rnd <= 0.70:
                child.keys[i] = pygame.K_a
            elif rnd <= 0.90:
                child.keys[i] = pygame.K_d
            else:
                child.keys[i] = None

def move_car(bot_car, key):
    moved = False
        
    if key == pygame.K_w and bot_car.fw:
        moved = True
        bot_car.bw = False
        bot_car.move_forward()

    if key == pygame.K_s and bot_car.bw:
        moved = True
        bot_car.fw = False
        bot_car.move_backward()

    if bot_car.vel > 0 and not moved:
        bot_car.reduce_speed_forward()
        bot_car.fw = True
        bot_car.bw = True
        
    if bot_car.vel < 0 and not moved:
        bot_car.reduce_speed_backward()
        bot_car.fw = True
        bot_car.bw = True

    if bot_car.vel > 0.7:
        moved = True

    if key == pygame.K_a and moved:
        bot_car.rotate(left=True)

    if key == pygame.K_d and moved:
        bot_car.rotate(right=True)

def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class AbstractCar:
    
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.fw = True
        self.bw = True

    def rotate(self, left=False, right=False):
        if left:
            self.angle += (self.rotation_vel * max((1 - self.vel/self.max_vel), 0.7))
        if right:
            self.angle -= (self.rotation_vel * max((1 - self.vel/self.max_vel), 0.7))

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), 90 + self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel)
        self.move()

    def reduce_speed_forward(self):
        self.vel = max(self.vel - self.acceleration * 0.5, 0)
        self.move()

    def reduce_speed_backward(self):
        self.vel = min(self.vel + self.acceleration * 0.5, 0)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

class BotCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (165, 200)


population = []
for _ in range(101):
    population.append(Individual())
population.remove(population[0])
indv = population[0]
bot_car = BotCar(4, 4)
new_population = population.copy()
    
i = 0
j = 0

p1 = (165, 200)
p2 = (0, 0)

run = True
first_car = True

while run:
    clock.tick(FPS)

    if i == 0 and not first_car:
        bot_car.img = GREEN_CAR
    else:
        bot_car.img = RED_CAR

    draw(WIN, images, bot_car)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            f = open("end.txt", "w")
            for tmp in population:
                f.write(" ".join(str(w_key) for w_key in tmp.keys) + "\n")
            f.close()
            break

    move_car(bot_car, indv.keys[j])

    j += 1
    next = False

    p2 = (bot_car.x, bot_car.y)
    indv.fitness += calculate_distance(p1, p2)
    p1 = p2

    if bot_car.collide(TRACK_BORDER_MASK) != None:
        next = True
    
    #new individual
    if j % len(indv.keys) == 0 or next:
        #print(indv.fitness)
        i += 1
        first_car = False
        #new generation
        if i % len(population) == 0:
            population.sort(key = lambda x : x.fitness, reverse=True)
            new_population = population.copy()

            print("Najbolji: " + str(population[0].fitness))
            '''print("-------Population--------")
            for index in range(len(population)):
                print(population[index].fitness)
            print("-------Population--------")'''
            
            for k in range(12, len(population), 2):
                par1_idx = selection(population, 30, None)
                par2_idx = selection(population, 30, par1_idx)

                crossover(population[par1_idx], population[par2_idx], new_population[k], new_population[k+1])

                mutation(new_population[k])
                mutation(new_population[k+1])

            population[:] = new_population[:]
            for index in range(len(population)):
                population[index].fitness = 0
            i = 0
        j = 0
        indv = population[i]
        bot_car = BotCar(4, 4)
        p1 = (165, 200)


    '''keys = pygame.key.get_pressed()

    moved = False
        
    if keys[pygame.K_w] and player_car.fw:
        moved = True
        player_car.bw = False
        player_car.move_forward()
        
    if keys[pygame.K_s] and player_car.bw:
        moved = True
        player_car.fw = False
        player_car.move_backward()
        
    if player_car.vel > 0 and not moved:
        player_car.reduce_speed_forward()
        player_car.fw = True
        player_car.bw = True
        
    if player_car.vel < 0 and not moved:
        player_car.reduce_speed_backward()
        player_car.fw = True
        player_car.bw = True

    if player_car.vel > 0.7:
        moved = True

    if keys[pygame.K_a] and moved:
        player_car.rotate(left=True)
        
    if keys[pygame.K_d] and moved:
        player_car.rotate(right=True)'''
    
pygame.quit()