# Fire Detection UAS

This is a research project inspired in one of the biggest problems that Galicia faces: forest fires.

## Motivation

Galicia is a region in the north of Spain which, due to its climate conditions, presents a very green 
landscape full of vegetation. Every year during the high temperature seasons a large amount of wildfires 
take place, consuming big areas of the forests and being a real danger for the inhabitants of the
surroundings. The causes of these fires are controversial. However, factors such as temperature,
humidity, wind and soil fuel can help to determine the fire risk index and predict areas in danger.
Wild fires have a huge impact on the environment and also severe long term consecuencies.
Galicia is not the only region that suffers from this, the Amazon rainforest has also registered a very
large number of wildfires this year (2019).


## Introduction

We mean by a UAV (Unmanned Aerial System) a fleet of UAV (Unmanned Aerial Vehicles, aka drones) that
are able to fly autonomously. The main purpose of this system is to detect a forest fire at a very early
stage, so it can be controlled and extinguished. The focus of this project is the software side, the
automation and coordination of the UAVs, so we assume they are equipped with the required payload for
fire detection under a certain radious.

We present two approaches; the first one is based on a field of forces that steers the vehicles towards the
gradient of the fire risk index, the second schedules routes from a base to visit the hottest spots.

## Approach 1: Surfers

Above we show the IRDI map (Indice the Riesgo De Fuego) on the 13th of October of 2019.

![](captures/irdi_map_2019_10_11.jpg)

Green and blue represent low and very low risk of fire, orange and red represent high and very high risk.

The first step in our process is to divide our space in cells with certain cell size.

![](captures/divided_irdi_map_2019_10_11.jpg)
