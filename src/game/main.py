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
import generator as Generator
import overlay as Overlay

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
PLAYER_HEALTH = 4
INIT_SCORE = 0


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
start_time = None


def start_game():
    global start_time

    start_time = pygame.time.get_ticks()


def stop_game():
    game_over()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() # Quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if event.key == pygame.K_r:
                    main()  # Restart
                    waiting = False


def game_over():
    font = pygame.font.SysFont("arial", 60)
    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    restart_text = font.render("ESC = Exit R = Restart", True, (255, 255, 255))
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 100))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
    pygame.display.update()


def main():

    global start_time
    start_game()

    player1 = Player(100, SCREEN_HEIGHT // 2, (0, 0, 255),
                     'Player1')  # TODO Player Parameter Egal weil ja detektion außer name
    player2 = Player(SCREEN_WIDTH - 100, SCREEN_HEIGHT // 2, (255, 0, 0), 'Player2')

    players = pygame.sprite.Group(player1, player2)

    fruits = pygame.sprite.Group()
    bombs = pygame.sprite.Group()

    generator = Generator.ObjectGenerator(
        initial_fruit_interval=4.5,
        initial_bomb_interval=5.5,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT
    )

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
        new_fruit = generator.generate_fruit()
        if new_fruit:
            fruits.add(new_fruit)

        new_bomb = generator.generate_bomb(random.choice(list(players)))
        if new_bomb:
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
                if bomb.player == player and player.rect.colliderect(bomb.rect):
                    bomb.player.catch_bomb(bomb)

        for player in players:
            if player.health <= 0:
                stop_game()

        # Früchte und Bomben zeichnen
        fruits.draw(screen)
        bombs.draw(screen)
        players.draw(screen)

        # render score & hp
        Overlay.draw_score(screen, players)
        Overlay.draw_hp(screen, players)
        Overlay.draw_game_time(screen, start_time)

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
