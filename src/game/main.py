"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""
import random

import cv2
import numpy as np
import pygame
import objects as Objects

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
PLAYER_HEALTH = 4
INIT_SCORE = 0

HEART = pygame.image.load('../../assets/sprites/heart.png')


# --------------------------------------------------------------------------
# -- player class
# --------------------------------------------------------------------------
class Player(pygame.sprite.Sprite):
    # -----------------------------------------------------------
    # init class

    def __init__(self, posX, posY, color, name):
        super(Player, self).__init__()
        self.surf = pygame.Surface((100, 30))
        # fill with color
        self.surf.fill(color)
        self.rect = self.surf.get_rect(center=(posX, posY))
        # start at screen center
        self.image = self.surf
        self.health = PLAYER_HEALTH
        self.score = INIT_SCORE
        self.name = name

    # -----------------------------------------------------------
    # update player rectangle
    def update(self, keys):
        if keys[pygame.K_UP]:
            self.rect.y -= 25
        if keys[pygame.K_DOWN]:
            self.rect.y += 25
        if keys[pygame.K_LEFT]:
            self.rect.x -= 25
        if keys[pygame.K_RIGHT]:
            self.rect.x += 25

    def catch_fruit(self, fruit):
        self.score += fruit.base_value + int(fruit.base_value * fruit.multiplier)
        fruit.kill()

    def catch_bomb(self, bomb):
        bomb.kill()


# --------------------------------------------------------------------------
# -- game
# --------------------------------------------------------------------------

# init pygame
pygame.init()

# set display size and caption
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Computer Vision Game")

# init game clock
fps = 30
clock = pygame.time.Clock()

# opencv - init webcam capture
cap = cv2.VideoCapture(0)
# set width & height to screen size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())


# init player
# player = Player(screen.get_width()/2, screen.get_height()/2)

# -------------
def render_score(players):
    for player in players:
        font = pygame.font.SysFont("arial", 26)
        score_text = font.render(f'{player.name}   Score: {player.score}', True, (0, 0, 0))
        if player.name == 'Player1':
            # Score und Herzen für Spieler 1 (oben links)
            screen.blit(score_text, (20, 20))
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (20 + i * 35, 50))

        else:
            # Score und Herzen für Spieler 2 (oben rechts)
            screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 20, 20))
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (SCREEN_WIDTH - 20 - (i + 1) * 35, 50))


def main():
    player1 = Player(100, SCREEN_HEIGHT // 2, (0, 0, 255), 'Player1') # TODO Player Parameter Egal weil ja detektion außer name
    player2 = Player(SCREEN_WIDTH - 100, SCREEN_HEIGHT // 2, (255, 0, 0), 'Player2')

    players = pygame.sprite.Group(player1, player2)

    fruits = pygame.sprite.Group()
    bombs = pygame.sprite.Group()

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # -- opencv & viz image
        ret, cameraFrame = cap.read()
        imgRGB = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2RGB)
        # image needs to be rotated for pygame
        imgRGB = np.rot90(imgRGB)
        # convert image to pygame and visualize
        gameFrame = pygame.surfarray.make_surface(imgRGB).convert()
        screen.blit(gameFrame, (0, 0))

        # --------------------------------------------------------------
        keys = pygame.key.get_pressed()

        for player in players:
            player.update(keys)

        # Früchte und Bomben generator
        if random.randint(1, 100) > 98:
            depth = random.uniform(0.6, 0.9)

            new_fruit = Objects.Fruit(random.randint(0, SCREEN_WIDTH), 0,
                                      random.choice(list(Objects.FRUIT_SPRITES.keys())),
                                      random.uniform(1.0, 10.0), random.randint(5, 20), depth=depth)
            fruits.add(new_fruit)

        if random.randint(1, 200) > 199:
            depth = random.uniform(0.5, 0.7)
            new_bomb = Objects.Bomb(random.randint(0, SCREEN_WIDTH), 0,
                                    random.choice(list(Objects.BOMB_SPRITES.keys())),
                                    player2, random.uniform(5, 20), random.uniform(2.0, 10.0), depth=depth)

            bombs.add(new_bomb)

        # Früchte und Bomben bewegen updaten etc.
        fruits.update()
        bombs.update()

        # Frucht und Spieler
        for fruit in fruits:
            for player in players:
                if player.rect.colliderect(fruit.rect):
                    player.catch_fruit(fruit)

        # Bombe und Spieler
        for bomb in bombs:
            for player in players:
                if player.rect.colliderect(bomb.rect):
                    bomb.player.catch_bomb(bomb)

        # Früchte und Bomben zeichnen
        fruits.draw(screen)
        bombs.draw(screen)
        players.draw(screen)

        # render score & hp
        render_score(players)

        # update  screen
        pygame.display.update()
        # set clock
        clock.tick(fps)

    # quit game
    pygame.quit()

    # release capture
    cap.release()


if __name__ == "__main__":
    main()


# TODO Game Over Screen
# TODO Main Menue
# TODO Bomben Spielerspezifisch