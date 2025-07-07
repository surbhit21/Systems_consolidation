/*
 *  ANNarchy-version: 5.0.0rc2
 */
#pragma once

#include "ANNarchy.hpp"
#include "helper_functions.hpp"
#include "LILMatrix.hpp"




extern PopStruct0 *pop0;
extern PopStruct0 *pop0;
extern double dt;
extern long int t;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: E_pop -> E_pop with target exc
/////////////////////////////////////////////////////////////////////////////
extern struct ProjStruct0 *proj0;
struct ProjStruct0 : LILMatrix<int, int> {
    ProjStruct0() : LILMatrix<int, int>( 70, 70) {
        // HACK: the object constructor is now called by nanobind, need to update reference in C++ library
        proj0 = this;

    #ifdef _TRACE_INIT
        std::cout << "  ProjStruct0 - this = " << this << " has been allocated." << std::endl;
    #endif
    }


    bool init_from_lil( std::vector<int> row_indices,
                        std::vector< std::vector<int> > column_indices,
                        std::vector< std::vector<double> > values,
                        std::vector< std::vector<int> > delays,
                        bool requires_sorting) {
        // The LIL entries are not sorted which might lead to worse psp access patterns
        if (requires_sorting) {
        #ifdef _DEBUG
            std::cout << "   ... sort the LIL entries by row index ..." << std::endl;
        #endif
            auto tmp = std::vector<int>(row_indices.size());
            std::iota(tmp.begin(), tmp.end(), 0);

            // sort row indices
            pairsort<int, int>(row_indices.data(), tmp.data(), row_indices.size());

            // apply sorting to column_indices
            auto new_column_indices= std::vector< std::vector<int> >();
            for(int i = 0; i < row_indices.size(); i++) {
                new_column_indices.push_back(column_indices[tmp[i]]);
            }
            column_indices = std::move(new_column_indices);

            // apply sorting to values
            auto new_values = std::vector< std::vector<double> >();
            for(int i = 0; i < row_indices.size(); i++) {
                new_values.push_back(values[tmp[i]]);
            }
            values = std::move(new_values);

            // apply sorting to delays
            auto new_delays = std::vector< std::vector<int> >();
            for(int i = 0; i < row_indices.size(); i++) {
                new_delays.push_back(delays[tmp[i]]);
            }
            delays = std::move(new_delays);
        }

        bool success = static_cast<LILMatrix<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local variable w
        w = init_matrix_variable<double>(static_cast<double>(0.0));
        update_matrix_variable_all<double>(w, values);


        // init other variables than 'w' or delay
        if (!init_attributes()){
            return false;
        }

    #ifdef _DEBUG_CONN
        static_cast<LILMatrix<int, int>*>(this)->print_data_representation();
    #endif
        return true;
    }





    // Transmission and plasticity flags
    bool _transmission, _axon_transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Global parameter tau_w
    double  tau_w ;

    // Global parameter lr
    double  lr ;

    // Global parameter tau_decay
    double  tau_decay ;

    // Global parameter max_weight
    double  max_weight ;

    // Global parameter min_weight
    double  min_weight ;

    // Local variable w
    std::vector< std::vector<double > > w;

    // Semiglobal parameter freeze
    std::vector< bool >  freeze ;




    // Method called to allocate/initialize the variables
    bool init_attributes() {

        // Global parameter tau_w
        tau_w = 0.0;

        // Global parameter lr
        lr = 0.0;

        // Global parameter tau_decay
        tau_decay = 0.0;

        // Global parameter max_weight
        max_weight = 0.0;

        // Global parameter min_weight
        min_weight = 0.0;

        // Semiglobal parameter freeze
        freeze = init_vector_variable<bool>(static_cast<bool>(false));




        return true;
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _TRACE_INIT
        std::cout << "  ProjStruct0::init_projection(post_size = " << pop0->size << ", pre_size = " << pop0->size << ") - this = " << this << std::endl;
    #endif

        _transmission = true;
        _axon_transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        init_attributes();



    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::compute_psp()" << std::endl;
    #endif
        double sum;

        if (_transmission && pop0->_active) {



            for (int i = 0; i < post_rank.size(); i++) {

                sum = 0.0;
                for (int j = 0; j < pre_rank[i].size(); j++) {
                    sum += pop0->r[pre_rank[i][j]]*w[i][j] ;
                }
                pop0->_sum_exc[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::update_synapse()" << std::endl;
    #endif

        int rk_post, rk_pre;
        double _dt = dt * _update_period;

        // Check periodicity
        if(_transmission && _update && pop0->_active && ( (t - _update_offset)%_update_period == 0L) ){
            // Global variables


            // Semiglobal/Local variables
            for (int i = 0; i < post_rank.size(); i++) {
                rk_post = post_rank[i]; // Get postsynaptic rank

                // Semi-global variables


                // Local variables
                for (int j = 0; j < pre_rank[i].size(); j++) {
                    rk_pre = pre_rank[i][j]; // Get presynaptic rank

                    // dw/_dt = if freeze: 0 else: (lr*(pre.r * post.r)/tau_w - w/tau_decay)
                    double _w = (freeze[i] ? 0 : pop0->r[rk_post]*pop0->r[rk_pre]*lr/tau_w - w[i][j]/tau_decay);

                    // dw/_dt = if freeze: 0 else: (lr*(pre.r * post.r)/tau_w - w/tau_decay)
                    if(_plasticity){
                    w[i][j] += _dt*_w ;
                    if(w[i][j] < min_weight)
                        w[i][j] = min_weight;
                    if(w[i][j] > max_weight)
                        w[i][j] = max_weight;

                    }

                }
            }
        }

    }

    // Post-synaptic events
    void post_event() {

    }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all_double(std::string name) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_all_double(name = "<<name<<")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_all<double>(w);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row_double(std::string name, int rk_post) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_row_double(name = "<<name<<", rk_post = "<<rk_post<<")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_row<double>(w, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_row_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk_post, int rk_pre) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_double(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_double(std::string name, std::vector<std::vector<double>> value) {
    #ifdef _DEBUG
        auto min_value = std::numeric_limits<double>::max();
        auto max_value = std::numeric_limits<double>::min();
        for (auto it = value.cbegin(); it != value.cend(); it++ ){
            auto loc_min = *std::min_element(it->cbegin(), it->cend());
            if (loc_min < min_value)
                min_value = loc_min;
            auto loc_max = *std::max_element(it->begin(), it->end());
            if (loc_max > max_value)
                max_value = loc_max;
        }
        std::cout << "ProjStruct0::set_local_attribute_all_double(name = " << name << ", min(" << name << ")=" <<std::to_string(min_value) << ", max("<<name<<")="<<std::to_string(max_value)<< ")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

            return;
        }

    }

    void set_local_attribute_row_double(std::string name, int rk_post, std::vector<double> value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::set_local_attribute_row_double(name = "<<name<<", rk_post = " << rk_post << ", min("<<name<<")="<<std::to_string(*std::min_element(value.begin(), value.end())) << ", max("<<name<<")="<<std::to_string(*std::max_element(value.begin(), value.end()))<< ")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

            return;
        }

    }

    void set_local_attribute_double(std::string name, int rk_post, int rk_pre, double value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::set_local_attribute_double(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<", value = " << std::to_string(value) << ")" << std::endl;
    #endif

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

            return;
        }

    }

    double get_global_attribute_double(std::string name) {

        // Global parameter tau_w
        if ( name.compare("tau_w") == 0 ) {

            return tau_w;
        }

        // Global parameter lr
        if ( name.compare("lr") == 0 ) {

            return lr;
        }

        // Global parameter tau_decay
        if ( name.compare("tau_decay") == 0 ) {

            return tau_decay;
        }

        // Global parameter max_weight
        if ( name.compare("max_weight") == 0 ) {

            return max_weight;
        }

        // Global parameter min_weight
        if ( name.compare("min_weight") == 0 ) {

            return min_weight;
        }


        // should not happen
        std::cerr << "ProjStruct0::get_global_attribute_double: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute_double(std::string name, double value) {

        // Global parameter tau_w
        if ( name.compare("tau_w") == 0 ) {
            tau_w = value;

            return;
        }

        // Global parameter lr
        if ( name.compare("lr") == 0 ) {
            lr = value;

            return;
        }

        // Global parameter tau_decay
        if ( name.compare("tau_decay") == 0 ) {
            tau_decay = value;

            return;
        }

        // Global parameter max_weight
        if ( name.compare("max_weight") == 0 ) {
            max_weight = value;

            return;
        }

        // Global parameter min_weight
        if ( name.compare("min_weight") == 0 ) {
            min_weight = value;

            return;
        }

    }

    std::vector<bool> get_semiglobal_attribute_all_bool(std::string name) {

        // Semiglobal parameter freeze
        if ( name.compare("freeze") == 0 ) {

            return get_vector_variable_all<bool>(freeze);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_semiglobal_attribute_all_bool: " << name << " not found" << std::endl;
        return std::vector<bool>();
    }

    bool get_semiglobal_attribute_bool(std::string name, int rk_post) {

        // Semiglobal parameter freeze
        if ( name.compare("freeze") == 0 ) {

            return get_vector_variable<bool>(freeze, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_semiglobal_attribute_bool: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all_bool(std::string name, std::vector<bool> value) {

        // Semiglobal parameter freeze
        if ( name.compare("freeze") == 0 ) {
            update_vector_variable_all<bool>(freeze, value);

            return;
        }

    }

    void set_semiglobal_attribute_bool(std::string name, int rk_post, bool value) {

        // Semiglobal parameter freeze
        if ( name.compare("freeze") == 0 ) {
            update_vector_variable<bool>(freeze, rk_post, value);

            return;
        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILMatrix<int, int>*>(this)->size_in_bytes();

        // Local variable w
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it = w.cbegin(); it != w.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Global parameter tau_w
        size_in_bytes += sizeof(double);

        // Global parameter lr
        size_in_bytes += sizeof(double);

        // Global parameter tau_decay
        size_in_bytes += sizeof(double);

        // Global parameter max_weight
        size_in_bytes += sizeof(double);

        // Global parameter min_weight
        size_in_bytes += sizeof(double);

        // Semiglobal parameter freeze
        size_in_bytes += sizeof(std::vector<bool>);
        size_in_bytes += sizeof(bool) * freeze.capacity();

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::clear() - this = " << this << std::endl;
    #endif

        // Connectivity
        static_cast<LILMatrix<int, int>*>(this)->clear();

        // w
        for (auto it = w.begin(); it != w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        w.clear();
        w.shrink_to_fit();

        // freeze
        freeze.clear();
        freeze.shrink_to_fit();

    }
};

