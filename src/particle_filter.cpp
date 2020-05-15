/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles

  // This line creates a normal (Gaussian) distribution for x, y, and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
	// Create particle
    Particle sample_particle;
    
    // Set particle values
    sample_particle.id = i;
    sample_particle.x = dist_x(gen);
    sample_particle.y = dist_y(gen);
    sample_particle.theta = dist_theta(gen);
    sample_particle.weight = 1.0;

    // Update particle vector with particle
    particles.push_back(sample_particle);
    weights.push_back(1);
  }
  
  // Set flag to initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  // This line creates a normal (Gaussian) distribution for x, y, and theta
  for (int i = 0; i < num_particles; ++i)
  {
    // Calculation depends if yaw rate ~0 or !=0
    if (fabs(yaw_rate) < 0.001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // Add gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  } 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  float min_distance;
  float min_index;
  float obv_x;
  float obv_y;
  float distance;
  for(int i = 0; i < observations.size(); ++i)
  {
    obv_x = observations[i].x;
    obv_y = observations[i].y;
    min_distance = numeric_limits<float>::max();
    
    for(int j = 0; j < predicted.size(); ++j)
    {
      distance = dist(obv_x, obv_y, predicted[j].x, predicted[j].y);
      if (distance < min_distance)
      {
        min_index = j;
        min_distance = distance;
      }
    }
    observations[i].id = min_index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  vector<LandmarkObs> trans_observations(observations.size());
  vector<LandmarkObs> map_landmarks_in_range;
  float obv_x;
  float obv_y;
  float trans_x;
  float trans_y;
  
  // Transform all observations into map coordinates
  for (int i = 0; i < num_particles; ++i)
  {
    // Clear the landmarks for each particle iteration
    if (map_landmarks_in_range.size() > 0)
      map_landmarks_in_range.clear();
    
    for (int j = 0; j < observations.size(); ++j)
    {
      obv_x = observations[j].x;
      obv_y = observations[j].y;
      trans_x = particles[i].x + obv_x*cos(particles[i].theta) - obv_y*sin(particles[i].theta);
      trans_y = particles[i].y + obv_x*sin(particles[i].theta) + obv_y*cos(particles[i].theta);
      trans_observations[j].x = trans_x;
      trans_observations[j].y = trans_y;
    }

    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
    {
      float landmark_id = map_landmarks.landmark_list[k].id_i;
      float landmark_x = static_cast<double>(map_landmarks.landmark_list[k].x_f);
      float landmark_y = static_cast<double>(map_landmarks.landmark_list[k].y_f);
      if (dist(particles[i].x, particles[i].y, landmark_x, landmark_y) <= sensor_range)
      {
        LandmarkObs landmark = {landmark_id, landmark_x, landmark_y};
        map_landmarks_in_range.push_back(landmark);
      }
    }
    
    // Associate the landmarks' ID to the transformed observations
    dataAssociation(map_landmarks_in_range, trans_observations);
    
    // Pre-calculate variances in x and y
    float std_x_2 = pow(std_landmark[0], 2);
    float std_y_2 = pow(std_landmark[1], 2);
    
    // Re-initialize to 1 to multiply probabilities
    particles[i].weight = 1;
    
    // Recalculate the weight using the associated landmarks for each particle
    for (int p = 0; p < trans_observations.size(); ++p)
    {
      // Store observation coordinates
      float x = trans_observations[p].x;
      float y = trans_observations[p].y;
      
      // Obtain index of associated landmark (nearest neighbor) and get landmark coordinates
      int index = trans_observations[p].id;
      float mu_x = map_landmarks_in_range[index].x;
      float mu_y = map_landmarks_in_range[index].y;

      // Calculate powers
      float diff_x_2 = pow(x - mu_x, 2);
      float diff_y_2 = pow(y - mu_y, 2);
      
      // Calculate exponent
      float expo = exp(-(diff_x_2/(2*std_x_2) + diff_y_2/(2*std_y_2)));
      
      // Calculate denominator
      float denom = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
      
      // Combine for probaility of this landmark
      float prob = denom * expo;
      
      // Multiply probability
      particles[i].weight *= prob; 
    }
  }
}

void ParticleFilter::resample() {
  // Refresh weights vector with the newly current particle weights
  weights.clear();
  for (int i = 0; i < num_particles; ++i)
    weights.push_back(particles[i].weight);
  
  // Use default randomizing generator and a discrete distribution on the full range of weights
  default_random_engine gen;
  discrete_distribution<size_t> dist_gen(weights.begin(), weights.end());
  
  // Temporary particle vector to store resampled particles
  vector<Particle> resample(num_particles);
  
  // Resample all particles using random generator with weights
  for (int i = 0; i < num_particles; ++i)
    resample[i] = particles[dist_gen(gen)];
  
  // Assign resampled particles to original vector  
  particles = resample;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}