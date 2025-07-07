
#include "ANNarchy.hpp"



/*
 * Internal data
 *
 */
double dt;
long int t;
std::vector<std::mt19937> rng;

// Custom constants


// Populations
PopStruct0 *pop0;


// Projections


// Global operations


/*
 * Recorders
 */
std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder){
    int found = -1;

    for (unsigned int i=0; i<recorders.size(); i++) {
        if (recorders[i] == nullptr) {
            found = i;
            break;
        }
    }

    if (found != -1) {
        // fill a previously cleared slot
        recorders[found] = recorder;
        return found;
    } else {
        recorders.push_back(recorder);
        return recorders.size() - 1;
    }
}

Monitor* getRecorder(int id) {
    if (id < recorders.size())
        return recorders[id];
    else
        return nullptr;
}

void removeRecorder(Monitor* recorder){
    for (unsigned int i=0; i<recorders.size(); i++){
        if (recorders[i] == recorder) {
            // mark the slot as free
            recorders[i] = nullptr;
            break;
        }
    }
}

/*
 *  Simulation methods
 */
// Simulate a single step
void singleStep()
{
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- evaluate step " << t << " (" << t * dt << " ms) --" << std::endl;
#endif


    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////


#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update psp/conductances ..." << std::endl;
#endif



    ////////////////////////////////
    // Recording target variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Record psp/conductances ..." << std::endl;
#endif
    for (unsigned int i=0; i < recorders.size(); i++) {
        if (recorders[i])
            recorders[i]->record_targets();
    }

    ////////////////////////////////
    // Update random distributions
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Draw required random numbers ..." << std::endl;
#endif




    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate neural ODEs ..." << std::endl;
#endif

	pop0->update();



    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update delay queues ..." << std::endl;
#endif


    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update global operations ..." << std::endl;
#endif




    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate synaptic ODEs ..." << std::endl;
#endif




    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////




    ////////////////////////////////
    // Structural plasticity
    ////////////////////////////////


    ////////////////////////////////
    // Recording neural / synaptic variables
    ////////////////////////////////

    for (unsigned int i=0; i < recorders.size(); i++){
        if (recorders[i])
            recorders[i]->record();
    }


    ////////////////////////////////
    // Increase internal time
    ////////////////////////////////
    t++;



#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- simulation step " << t << " completed --" << std::endl;
#endif
}

// Simulate the network for the given number of steps,
// called from python
void run(const int nbSteps) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Perform simulation for " << nbSteps << " steps." << std::endl;
#endif

    // apply changes implied by structural plasticity (spike only)


    // perform the simulation
    for(int i=0; i<nbSteps; i++) {
        singleStep();
    }

}

// Simulate the network for a single steps,
// called from python
void step() {

    // apply changes implied by structural plasticity (spike only)


    // perform a single step (size dt)
    singleStep();

}

int run_until(const int steps, std::vector<int> populations, bool or_and)
{
    // apply changes implied by structural plasticity (spike only)


    // perform the simulation until the condition is satisfied

    run(steps);
    return steps;

}

/*
 *  Initialization methods
 */
// Initialize the internal data and the random numbers generator
void initialize(const double _dt) {


    // Internal variables
    dt = _dt;
    t = static_cast<long int>(0);

    // Populations
    // Initialize populations
    pop0->init_population();


    // Projections
    // Initialize projections


    // Custom constants


}

// Change the seed of the RNG
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq) {
#ifdef _DEBUG
    std::cout << "ANNarchyCore::setSeed(): " << seed << ", " << num_sources << ", " << std::string((use_seed_seq) ? "true" : "false") << std::endl;
#endif
    // sanity check
    if (num_sources > 1)
        std::cerr << "WARNING - ANNarchyCore::setSeed(): num_sources should be 1 for single thread code." << std::endl;

    rng.clear();

    rng.push_back(std::mt19937(seed));

    rng.shrink_to_fit();
}

/*
 *  Life-time management
 */
void create_cpp_instances() {
#if defined(_TRACE_INIT) || defined(_DEBUG)
    std::cout << "Instantiate C++ objects ..." << std::endl;
#endif
}

void destroy_cpp_instances() {
#if defined(_TRACE_INIT) || defined(_DEBUG)
    std::cout << "Destroy C++ objects ..." << std::endl;
#endif
}

/*
 * Access to time and dt
 */
long int getTime() {return t;}
void setTime(const long int t_) { t=t_;}
double getDt() { return dt;}
void setDt(const double dt_) { dt=dt_;}

/*
 * Number of threads
 *
*/
void setNumberThreads(const int threads, const std::vector<int> core_list)
{
    if (threads > 1) {
        std::cerr << "WARNING: a call of setNumberThreads() is without effect on single thread simulation code." << std::endl;
    }

    if (core_list.size()>1) {
        std::cerr << "The provided core list is ambiguous and therefore ignored." << std::endl;
        return;
    }

#ifdef __linux__
    // set a cpu mask to prevent moving of threads
    cpu_set_t mask;

    // no CPUs selected
    CPU_ZERO(&mask);

    // no proc_bind
    for(auto it = core_list.begin(); it != core_list.end(); it++)
        CPU_SET(*it, &mask);
    const int set_result = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
#else
    if (!core_list.empty()) {
        std::cout << "WARNING: manipulation of CPU masks is only available for linux systems." << std::endl;
    }
#endif
}
