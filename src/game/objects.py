import pygame


SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
SCREEN 	      = [SCREEN_WIDTH,SCREEN_HEIGHT]

FRUIT_SPRITES = {

    'appleG': pygame.image.load('../../assets/sprites/Apple_Green.png'),
    'appleR': pygame.image.load('../../assets/sprites/Apple_Red.png'),
    'appleY': pygame.image.load('../../assets/sprites/Apple_Yellow.png'),
    'banana': pygame.image.load('../../assets/sprites/Banana.png'),
    'berry': pygame.image.load('../../assets/sprites/Berry.png'),
    'cherry': pygame.image.load('../../assets/sprites/Cherry.png'),
    'lemon': pygame.image.load('../../assets/sprites/Lemon.png'),
    'lime': pygame.image.load('../../assets/sprites/Lime.png'),
    'orange': pygame.image.load('../../assets/sprites/Orange.png'),
    'pear': pygame.image.load('../../assets/sprites/Pear.png'),
    'plum': pygame.image.load('../../assets/sprites/Plum.png'),
    'watermelon': pygame.image.load('../../assets/sprites/Watermelon.png')

}

BOMB_SPRITES = {

    'green': pygame.image.load('../../assets/sprites/bombGreen.png'),
    'red': pygame.image.load('../../assets/sprites/bombRed.png')

}


class Fruit(pygame.sprite.Sprite):
    def __init__(self, x, y, fruit_type, multiplier, base_value, depth=1.0):
        super(Fruit, self).__init__()
        scaling = 1.0 / depth
        fruit_width = 100 * scaling
        fruit_height = 100 * scaling
        self.x = x
        self.y = y
        if x - fruit_width // 2 < 0:
            self.x = fruit_width // 2
        elif x + fruit_width // 2 > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - fruit_width // 2

        self.image = pygame.transform.scale(FRUIT_SPRITES[fruit_type], (fruit_width, fruit_height))
        self.rect = self.image.get_rect(center=(self.x, self.y))

        self.multiplier = multiplier
        self.base_value = base_value
        self.depth = depth

        self.font = pygame.font.Font(None, 24)
        self.render_multiplier()

    def update(self):
        fall_speed = 5 / self.depth
        self.rect.y += fall_speed
        if self.rect.y > SCREEN_HEIGHT:
            self.kill()

    def render_multiplier(self):

        multiplier_text = f"{self.multiplier:.1f}x"
        text_surface = self.font.render(multiplier_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.image.get_width() // 2, self.image.get_height() // 2))

        self.image.blit(text_surface, text_rect)


class Bomb(pygame.sprite.Sprite):
    def __init__(self, x, y, bomb_type, player, multiplier, depth=1.0):
        super(Bomb, self).__init__()
        scaling = 1.0 / depth
        bomb_width = 100 * scaling
        bomb_height = 100 * scaling
        self.x = x
        self.y = y
        if x - bomb_width // 2 < 0:
            self.x = bomb_width // 2
        elif x + bomb_width // 2 > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - bomb_width // 2

        self.image = pygame.transform.scale(BOMB_SPRITES[bomb_type], (bomb_width, bomb_height))
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.player = player
        self.multiplier = multiplier
        self.bomb_type = bomb_type
        self.depth = depth

        self.font = pygame.font.Font(None, 32)
        self.render_multiplier()

    def update(self):
        fall_speed = 5 / self.depth
        self.rect.y += fall_speed

        if self.rect.y > SCREEN_HEIGHT:
            self.kill()
            self.player.health -= 1
            self.player.score = int(self.player.score * (1 - self.multiplier))

    def render_multiplier(self):

        multiplier_text = f"-{self.multiplier:.1f}x"
        text_surface = self.font.render(multiplier_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.image.get_width() // 2, self.image.get_height() // 2))

        self.image.blit(text_surface, text_rect)
