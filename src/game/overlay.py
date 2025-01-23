import pygame

HEART = pygame.image.load('../../assets/sprites/heart.png')


def draw_score(screen, players):
    """
    Zeichnet die Punktzahl der aktiven Spieler auf den Bildschirm.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
            players (Group): Liste von Spielerobjekten mit Attributen name, score, und activated.
    """
    for player in players:
        font = pygame.font.SysFont("arial", 26)
        if player.name == 'Player1' and player.activated:
            score_text = font.render(f'{player.name}   Score: {player.score}', True, (0, 0, 255))
            screen.blit(score_text, (20, 20))
        elif player.name == 'Player2' and player.activated:
            score_text = font.render(f'{player.name}   Score: {player.score}', True, (255, 0, 0))
            screen.blit(score_text, (screen.get_width() - score_text.get_width() - 20, 20))


def draw_hp(screen, players):
    """
    Zeichnet die verbleibenden Lebenspunkte (HP) der Spieler in Form von Herzen.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
            players (Group): Liste von Spielerobjekten mit Attributen name, health und activated.
    """
    for player in players:
        if player.name == 'Player1' and player.activated:
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (20 + i * 35, 50))
        elif player.name == 'Player2' and player.activated:
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (screen.get_width() - 40 - (i + 1) * 35, 50))


def draw_game_time(screen, start_time):
    """
    Zeichnet den Spiel-Timer auf den Bildschirm.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
            start_time (float): Die Startzeit des Spiels in Millisekunden.
    """
    if start_time is not None:

        elapsed_time = pygame.time.get_ticks() - start_time

        seconds = elapsed_time // 1000  # Zeit in Sekunden umwandeln
        minutes = seconds // 60  # In Minuten
        seconds = seconds % 60  # Verbleibenden Sekunden

        # Format 00:00
        time_text = f"{minutes:02}:{seconds:02}"  # Min. 2 Stellen ansonsten mit 0 auffüllen

        # Rendern
        font = pygame.font.SysFont("arial", 48)
        time_surface = font.render(time_text, True, (0, 0, 0))

        # Position
        text_width = time_surface.get_width()
        center_x = (screen.get_width() - text_width) // 2
        center_y = 20

        # Zeichnen
        screen.blit(time_surface, (center_x, center_y))


def draw_game_over(screen, players):
    """
    Zeichnet den Game-Over-Bildschirm.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
    """
    font = pygame.font.SysFont("arial", 60)

    if players[0].score > players[1].score:
        winner = 'Player1'
    else:
        winner = 'Player2'

    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    winner_text = font.render(f"Winner: {winner}", True, (0, 255, 0))
    restart_text = font.render("R = Restart", True, (255, 255, 255))
    escape_text = font.render("ESC = Exit", True, (255, 255, 255))
    screen.blit(game_over_text, (screen.get_width() // 2 - game_over_text.get_width() // 2, screen.get_height() // 2 - 100))
    screen.blit(winner_text, (screen.get_width() // 2 - winner_text.get_width() // 2, screen.get_height() // 2))
    screen.blit(restart_text, (screen.get_width() // 2 - restart_text.get_width() // 2, screen.get_height() // 2 + 100))
    screen.blit(escape_text, (screen.get_width() // 2 - escape_text.get_width() // 2, screen.get_height() // 2 + 200))
    pygame.display.update()


def draw_start_screen(screen):
    """
    Zeichnet den Startbildschirm und wartet, bis der Spieler die Leertaste drückt.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
    """

    font_title = pygame.font.SysFont("arial", 72)
    font_button = pygame.font.SysFont("arial", 36)

    bg = pygame.image.load('../../assets/images/background.png')
    bg = pygame.transform.scale(bg, (screen.get_width(), screen.get_height()))
    screen.blit(bg, [0,0])
    title_text = font_title.render("Computer Vision Game", True, (0, 0, 0))
    screen.blit(title_text, (screen.get_width() // 2 - title_text.get_width() // 2, screen.get_height() // 3))
    play_button_text = font_button.render("Space to Start", True, (0, 0, 0))
    button_rect = pygame.Rect(screen.get_width() // 2 - 100, screen.get_height() // 2, 200, 60)
    pygame.draw.rect(screen, (0, 255, 255), button_rect)
    screen.blit(play_button_text, (button_rect.x + button_rect.width // 2 - play_button_text.get_width() // 2,
                                   button_rect.y + button_rect.height // 2 - play_button_text.get_height() // 2))
    pygame.display.flip()

    waiting = True

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False

def draw_countdown(screen, countdown_seconds=3):
    """
    Zeichnet einen Countdown auf den Bildschirm.
        Args:
            screen (Screen): Das pygame-Bildschirmobjekt.
            countdown_seconds (int): Die Dauer des Countdowns in Sekunden (Standard: 3).
    """

    font = pygame.font.SysFont("arial", 96)

    for i in range(countdown_seconds, 0, -1):
        bg = pygame.image.load('../../assets/images/background.png')
        bg = pygame.transform.scale(bg, (screen.get_width(), screen.get_height()))
        screen.blit(bg, [0, 0])
        countdown_text = font.render(str(i), True, (0, 0, 0))
        screen.blit(countdown_text, (screen.get_width() // 2 - countdown_text.get_width() // 2,
                                     screen.get_height() // 2 - countdown_text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(1000)