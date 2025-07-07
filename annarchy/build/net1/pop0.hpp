/*
 *  ANNarchy-version: 5.0.0rc2
 */
#pragma once

#include "ANNarchy.hpp"
#include <random>



extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;
extern double norm1_value(const double*, int);
extern double norm2_value(const double*, int);



///////////////////////////////////////////////////////////////
// Main Structure for the population of id 0 (E_pop)
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

    // Global parameter tau
    double  tau ;

    // Local parameter input_i
    std::vector< double > input_i;

    // Global parameter I0
    double  I0 ;

    // Global parameter I1
    double  I1 ;

    // Global parameter I2
    double  I2 ;

    // Local parameter Epsi_i
    std::vector< double > Epsi_i;

    // Local variable x
    std::vector< double > x;

    // Local variable r
    std::vector< double > r;

    // Local variable ex
    std::vector< double > ex;

    // Local variable inp
    std::vector< double > inp;

    // Local psp _sum_exc
    std::vector< double > _sum_exc;

    // Global operations
    double _norm1_x;
    double _norm2_x;

    // Random numbers





    // Access methods to the parameters and variables


    // Method called to initialize the data structures
    void init_population() {
    #ifdef _TRACE_INIT
        std::cout << "  PopStruct0::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Global parameter tau
        tau = 0.0;

        // Local parameter input_i
        input_i = std::vector<double>(size, 0.0);

        // Global parameter I0
        I0 = 0.0;

        // Global parameter I1
        I1 = 0.0;

        // Global parameter I2
        I2 = 0.0;

        // Local parameter Epsi_i
        Epsi_i = std::vector<double>(size, 0.0);

        // Local variable x
        x = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);

        // Local variable ex
        ex = std::vector<double>(size, 0.0);

        // Local variable inp
        inp = std::vector<double>(size, 0.0);

        // Initialize global operations
        _norm1_x = 0.0;
        _norm2_x = 0.0;

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

    if ( _active ){

            _norm1_x = norm1_value(x.data(), size);

            _norm2_x = norm2_value(x.data(), size);

    }
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

                // tau * dx/dt  + x = pos(input_i - I0 - I1 * norm1(x) - I2 * norm2(x) + Epsi_i + sum(exc))
                double _x = (-x[i] + positive(Epsi_i[i] - I0 - I1*_norm1_x - I2*_norm2_x + _sum_exc[i] + input_i[i]))/tau;

                // tau * dx/dt  + x = pos(input_i - I0 - I1 * norm1(x) - I2 * norm2(x) + Epsi_i + sum(exc))
                x[i] += dt*_x ;


                // r = if x < 1e-3: 0 else: x
                r[i] = (x[i] < 0.001 ? 0 : x[i]);


                // ex = Epsi_i
                ex[i] = Epsi_i[i];


                // inp = input_i
                inp[i] = input_i[i];


            }
        } // active

    }

    void spike_gather() {

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// tau
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * input_i.capacity();	// input_i
        size_in_bytes += sizeof(double);	// I0
        size_in_bytes += sizeof(double);	// I1
        size_in_bytes += sizeof(double);	// I2
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Epsi_i.capacity();	// Epsi_i
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * x.capacity();	// x
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * ex.capacity();	// ex
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * inp.capacity();	// inp
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
        input_i.clear();
        input_i.shrink_to_fit();
        Epsi_i.clear();
        Epsi_i.shrink_to_fit();

        // Variables
        x.clear();
        x.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();
        ex.clear();
        ex.shrink_to_fit();
        inp.clear();
        inp.shrink_to_fit();

        // RNGs

    }
};

