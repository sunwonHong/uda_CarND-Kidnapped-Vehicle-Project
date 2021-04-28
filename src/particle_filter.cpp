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

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

bool start = true;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    num_particles = 10;
	
	normal_distribution<double> d_x(x, std[0]);
	normal_distribution<double> d_y(y, std[1]);
	normal_distribution<double> d_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
	  Particle new_part;
      new_part.id = i;
	  new_part.x = d_x(gen);
	  new_part.y = d_y(gen);
	  new_part.theta = d_theta(gen);
	  new_part.weight = 0.1;
      particles.push_back(new_part);
	  weights.push_back(new_part.weight);
	}
  
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
	
	for (int i = 0; i < num_particles; i++) {
	  double x_part = particles[i].x;
	  double y_part = particles[i].y;
	  double theta_part = particles[i].theta;
	  
      double yaw_rate_saved;
	 
	  double x_temp;
	  double y_temp;
	  double theta_temp;
      
      if(yaw_rate != 0) {
	    x_temp = x_part + (velocity/yaw_rate) * (sin(theta_part + (yaw_rate * delta_t)) - sin(theta_part));
	    y_temp = y_part + (velocity/yaw_rate) * (cos(theta_part) - cos(theta_part + (yaw_rate * delta_t)));
	    theta_temp = theta_part + (yaw_rate * delta_t);
        yaw_rate_saved = yaw_rate;
//         std::cout<< "theta_part =" << theta_part << std::endl;
//         std::cout<< "velocity =" << velocity << std::endl;
//         std::cout<< "x_temp =" << x_part << std::endl;
//         std::cout<< "y_temp =" << y_part << std::endl;
      }
      else{
        x_temp = x_part + velocity * (sin(theta_part + (yaw_rate * delta_t)) - sin(theta_part));
        y_temp = y_part + velocity * (cos(theta_part) - cos(theta_part + (yaw_rate * delta_t)));
        theta_temp = theta_part + (yaw_rate * delta_t);
      }
	  
	  normal_distribution<double> d_x(x_temp, std_pos[0]);
	  normal_distribution<double> d_y(y_temp, std_pos[1]);
	  normal_distribution<double> d_theta(theta_temp, std_pos[2]);
	  
	  particles[i].x = d_x(gen);
	  particles[i].y = d_y(gen);
	  particles[i].theta = d_theta(gen);
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
    for (int j = 0; j < observations.size(); j++){
      LandmarkObs o = observations[j];
      double min_dist = 10000000;
      for(int i = 0; i < predicted.size(); i++){
        LandmarkObs p = predicted[i];
        if(dist(o.x, o.y, p.x, p.y) < min_dist){
          min_dist = dist(o.x, o.y, p.x, p.y);
          observations[j].id = p.id;
        }
      }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
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
  double sum = 0.0;

  for (int i = 0; i < num_particles; i++) {
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta_part = particles[i].theta;

    vector<LandmarkObs> l_trans;
    vector<LandmarkObs> obs_trans;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double l_x = map_landmarks.landmark_list[j].x_f;
      double l_y = map_landmarks.landmark_list[j].y_f;
      int l_id = map_landmarks.landmark_list[j].id_i;
      if (fabs(x_part - l_x) <= sensor_range && fabs(y_part - l_y) <= sensor_range) {
        l_trans.push_back(LandmarkObs{l_id, l_x, l_y});
      }
    }
    
    for (int j = 0; j < observations.size(); j++) {
      double t_x = x_part + cos(theta_part) * observations[j].x - sin(theta_part) * observations[j].y;
      double t_y = y_part + sin(theta_part) * observations[j].x + cos(theta_part) * observations[j].y;
      int t_id = observations[j].id;
      obs_trans.push_back(LandmarkObs{t_id, t_x, t_y});
    }

    dataAssociation(l_trans, obs_trans);
    particles[i].weight = 1.0;

    for (int j = 0; j < obs_trans.size(); j++) {
      double obs_temp_x = obs_trans[j].x;
      double obs_temp_y = obs_trans[j].y;
      int obs_temp_id = obs_trans[j].id;

      for (int k = 0; k < l_trans.size(); k++) {
        if (obs_temp_id == l_trans[k].id) {
          double l_temp_x = l_trans[k].x;
          double l_temp_y = l_trans[k].y;
          double mwith_weight = (1/(2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-1 * ((pow(obs_temp_x - l_temp_x, 2)/(2 * pow(std_landmark[0], 2))) + (pow(obs_temp_y - l_temp_y, 2)/(2 * pow(std_landmark[1], 2)))));
          particles[i].weight = particles[i].weight * mwith_weight;
          sum = sum + particles[i].weight;
        }
      }
    }
    weights[i] = particles[i].weight / sum;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	vector<Particle> part_new;
	uniform_int_distribution<int> dist_index(0, num_particles - 1);
	int ind = dist_index(gen);
	
	double beta = 0.0;
	
	double mw = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> dist_random(0.0, mw);
	double random = dist_random(gen);
  
	for (int i = 0; i < num_particles; i++) {
	  beta = beta + random * 2.0 * mw;
	  while (beta > weights[ind]) {
	    beta = beta - weights[ind];
	    ind = (ind + 1) % num_particles;
	  }
	  part_new.push_back(particles[ind]);
	}
	particles = part_new;
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
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
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

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}