import pygame

HEART = pygame.image.load('../../assets/sprites/heart.png')


def draw_score(screen, players):  # Score des jewl. Spielers zeichnen
    for player in players:
        font = pygame.font.SysFont("arial", 26)
        if player.name == 'Player1':
            score_text = font.render(f'{player.name}   Score: {player.score}', True, (0, 0, 255))
            screen.blit(score_text, (20, 20))
        else:
            score_text = font.render(f'{player.name}   Score: {player.score}', True, (255, 0, 0))
            screen.blit(score_text, (screen.get_width() - score_text.get_width() - 20, 20))


def draw_hp(screen, players):  # Hp des jewl. Spielers zeichnen

    for player in players:
        if player.name == 'Player1':
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (20 + i * 35, 50))
        else:
            for i in range(player.health):
                screen.blit(pygame.transform.scale(HEART, (60, 60)), (screen.get_width() - 40 - (i + 1) * 35, 50))


def draw_game_time(screen, start_time):  # Spieltimer zeichnen
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


def draw_game_over(screen):  # Game Over Screen zeichnen
    font = pygame.font.SysFont("arial", 60)
    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    restart_text = font.render("R = Restart", True, (255, 255, 255))
    escape_text = font.render("ESC = Exit", True, (255, 255, 255))
    screen.blit(game_over_text, (screen.get_width() // 2 - game_over_text.get_width() // 2, screen.get_height() // 2 - 100))
    screen.blit(restart_text, (screen.get_width() // 2 - restart_text.get_width() // 2, screen.get_height() // 2 + 50))
    screen.blit(escape_text, (screen.get_width() // 2 - escape_text.get_width() // 2, screen.get_height() // 2 + 150))
    pygame.display.update()
