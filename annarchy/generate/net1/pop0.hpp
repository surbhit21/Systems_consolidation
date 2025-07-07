/*
 *  ANNarchy-version: 5.0.0rc2
 */
#pragma once

#include "ANNarchy.hpp"
#include <random>



extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;



///////////////////////////////////////////////////////////////
// Main Structure for the population of id 0 (pop)
///////////////////////////////////////////////////////////////
extern struct PopStruct0 *pop0;
struct PopStruct0{

    PopStruct0(int size, int max_delay) {
        this->size = size;
        this->max_delay = max_delay;

        // HACK: the object constructor is now called by nanobind, need to update reference in C++ library
        pop0 = this;

    #ifdef _TRACE_INIT
        std::cout << "  PopStruct0 - this = " << this << " has been allocated." << std::endl;
    #endif
    }

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }



    // Neuron specific parameters and variables

    // Global parameter Iext
    double  Iext ;

    // Local variable r
    std::vector< double > r;

    // Local psp _sum_exc
    std::vector< double > _sum_exc;

    // Random numbers





    // Access methods to the parameters and variables


    // Method called to initialize the data structures
    void init_population() {
    #ifdef _TRACE_INIT
        std::cout << "  PopStruct0::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Global parameter Iext
        Iext = 0.0;

        // Local variable r
        r = std::vector<double>(size, 0.0);

        // Local psp _sum_exc
        _sum_exc = std::vector<double>(size, 0.0);






    }

    // Method called to reset the population
    void reset() {



    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update_rng()" << std::endl;
#endif

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update_global_ops()" << std::endl;
#endif

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update()" << std::endl;
#endif

        if( _active ) {
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    PopStruct0::update()" << std::endl;
        #endif

            // Updating the local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // dr/dt +r = pos(sum(exc) + Iext)
                double _r = -r[i] + positive(Iext + _sum_exc[i]);

                // dr/dt +r = pos(sum(exc) + Iext)
                r[i] += dt*_r ;


            }
        } // active

    }

    void spike_gather() {

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// Iext
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct0::clear() - this = " << this << std::endl;
#endif

            #ifdef _DEBUG
                std::cout << "PopStruct0::clear()" << std::endl;
            #endif
        // Parameters

        // Variables
        r.clear();
        r.shrink_to_fit();

        // RNGs

    }
};

