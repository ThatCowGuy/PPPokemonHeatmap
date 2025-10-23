# PPPokemonHeatmap

This project aims at analyzing Twitch VODs from the ongoing "Pi Plays Pokémon Sapphire" Stream (https://www.twitch.tv/winningsequence), where armityle29 attempts to beat Pokémon Sapphire through the transcendental constant Pi. In order to let Pi "play" the game, the digits 0-9 have been mapped to the controls of a Gameboy Advance and each second a new digit of Pi is being fed into the game. This experiment has been running for over 4 years by now, while achieving relatively little progression. Nonetheless, 4 years of streaming resulted in an enormous amount of data, which is highly interesting to analyze statistically.

My analysis aims to generate a Heatmap of Pi's Player Position throughout the journey. Since no positional data is being recorded during the streams, I am forced to reconstruct the data from the raw streaming material. First, selected stream VODs are downloaded before I apply some preprocessing steps to the footage in order to massively reduce the datasize without losing any information. In this preprocessing, the actual gaming footage is trimmed out, scaled down and grayscaled. I then reduce the framerate of the video material immensily, as I only need snapshots of the character's position roughly every second. These snapshots are then compared to low-resolution mosaic templates of the in-game maps that I created to match my preprocessing steps. In this comparison, I pinpoint the character's position by generating similarity maps and finding the best ranking match on them. Additionally, the search space is further decreased by using a Multi-Linked List system to keep track of the currently active map and it's available adjacent maps, and the proccess is parallelized through the threading package. 

While processing a stream VOD, I only keep track of changes in the player's position (as there are many, many reasons for the position to stagnate for longer periods of time), which I can then export into a CSV file. Using this recorded list of positions, I can then overlay a tracker onto my mosaic maps in order to visualize the player's path without any interruptions:

<div align="center">
  <img src="https://github.com/user-attachments/assets/c894ff2c-162d-417d-9fc4-b7822cc4778c" />
</div>


In the future, I intend to collect positional data from many stream VODs to process them into a single large CSV file, which I can then parse to generate an accurate heatmap of the player's position in order to analyze their erratic behaviour.
