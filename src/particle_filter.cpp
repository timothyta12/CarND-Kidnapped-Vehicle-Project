/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
  	// Set number of particles for the filter
  	num_particles = 1000;
  
  	// Create guassian noise distributions
  	default_random_engine gen;
  	normal_distribution<double> dist_x(x, std[0]);
  	normal_distribution<double> dist_y(y, std[1]);
  	normal_distribution<double> dist_theta(theta, std[2]);
  
  	// Create random particles using guassian noise around initial estimate
  	for (int i=0; i<num_particles; i++) {
      particles.push_back(Particle());
      particles[i].id = i;
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
      particles[i].weight = 1.0;
      weights.push_back(particles[i].weight);
    }
  
  	// Initialized filter
  	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  	default_random_engine gen;
  	normal_distribution<double> dist_x(0, std_pos[0]);
  	normal_distribution<double> dist_y(0, std_pos[1]);
  	normal_distribution<double> dist_theta(0, std_pos[2]);
  
	for (int i=0; i<num_particles; i++) {
      double theta_i = particles[i].theta;
      double theta_f = theta_i + yaw_rate*delta_t;
      
      particles[i].x += (velocity/yaw_rate)*(sin(theta_f) - sin(theta_i)) + dist_x(gen);
      particles[i].y += (velocity/yaw_rate)*(cos(theta_i) - cos(theta_f)) + dist_y(gen);
      particles[i].theta = theta_f + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i=0; i<observations.size(); i++) {
      double min_distance = std::numeric_limits<double>::max();
      LandmarkObs closest;
      for (unsigned int j=0; j<predicted.size(); j++) {
        double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        if (distance < min_distance) {
          closest = predicted[j];
          min_distance = distance;
        }
      }
      observations[i] = closest;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  for (int i=0; i<num_particles; i++) {
    Particle p = particles[i];
    
    // Transform observations from particle coordinates to global map coordinates
    vector<LandmarkObs> transformed_observations;
    for (unsigned int j=0; j<observations.size(); j++) {
      double t_x = p.x + cos(p.theta) * observations[j].x - sin(p.theta) * observations[j].y;
      double t_y = p.y + sin(p.theta) * observations[j].x + cos(p.theta) * observations[j].y;

      transformed_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }
    
    // Collect predicted landmarks
    vector<LandmarkObs> predictions;
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) {
      // Landmark in sensor range
      if (dist(p.x, map_landmarks.landmark_list[j].x_f, p.y, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
        LandmarkObs landmark = {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f};
        predictions.push_back(landmark);
      }
    }
    
    // Binds Landmark to Observations
    dataAssociation(predictions, transformed_observations);
    
    particles[i].weight = 1.0;
    for (unsigned j=0; j<transformed_observations.size(); j++) {
      double o_x = transformed_observations[j].x;
      double o_y = transformed_observations[j].y;
      
      for (unsigned k=0; k<predictions.size(); k++) {
        double pr_x = predictions[k].x;
        double pr_y = predictions[k].y;
        
        if (transformed_observations[j].id == predictions[k].id) {
          double s_x = std_landmark[0];
          double s_y = std_landmark[1];
          double obs_w = (1 / (2*M_PI*s_x*s_y)) * exp(-( (pow(pr_x - o_x, 2)/(2*pow(s_x, 2))) + (pow(pr_y - o_y, 2)/(2*pow(s_y, 2))) ) ); 
          particles[i].weight *= obs_w;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  	for (int i=0; i<num_particles; i++) {
      weights[i] = particles[i].weight;
    }
  
    default_random_engine gen;
    discrete_distribution<int> dis(weights.begin(), weights.end());
  
  	vector<Particle> resampled_particles;
  	for (int i=0; i<num_particles; i++) {
      resampled_particles.push_back(particles[dis(gen)]);
    }
  	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
