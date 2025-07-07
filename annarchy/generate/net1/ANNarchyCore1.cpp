#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/bind_vector.h>

#include "ANNarchy.hpp"

NB_MODULE(ANNarchyCore1, m) {

    // Global functions
    m.def("set_seed", &setSeed);
    m.def("pyx_create", &create_cpp_instances);
    m.def("pyx_initialize", &initialize);
    m.def("run", &run);
    m.def("run_until", &run_until);
    m.def("step", &step);
    m.def("set_time", &setTime);
    m.def("get_time", &getTime);

    // Target device specific
	m.def("set_number_threads", &setNumberThreads);

    // Simulation-related objects





    // PopStruct0
    nanobind::class_<PopStruct0>(m, "pop0_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct0::size)
        .def_rw("max_delay", &PopStruct0::max_delay)

        // Attributes
		.def("update_max_delay", &PopStruct0::update_max_delay)
		.def_rw("tau", &PopStruct0::tau)
		.def_rw("input_i", &PopStruct0::input_i)
		.def_rw("I0", &PopStruct0::I0)
		.def_rw("I1", &PopStruct0::I1)
		.def_rw("I2", &PopStruct0::I2)
		.def_rw("Epsi_i", &PopStruct0::Epsi_i)
		.def_rw("x", &PopStruct0::x)
		.def_rw("r", &PopStruct0::r)
		.def_rw("ex", &PopStruct0::ex)
		.def_rw("inp", &PopStruct0::inp)
		.def_rw("_sum_exc", &PopStruct0::_sum_exc)


        // Other methods

        .def("activate", &PopStruct0::set_active)
        .def("reset", &PopStruct0::reset)
        .def("clear", &PopStruct0::clear);


    // PopStruct1
    nanobind::class_<PopStruct1>(m, "pop1_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct1::size)
        .def_rw("max_delay", &PopStruct1::max_delay)

        // Attributes
		.def("update_max_delay", &PopStruct1::update_max_delay)
		.def_rw("tau", &PopStruct1::tau)
		.def_rw("min_fr", &PopStruct1::min_fr)
		.def_rw("max_fr", &PopStruct1::max_fr)
		.def_rw("r", &PopStruct1::r)
		.def_rw("_sum_exc", &PopStruct1::_sum_exc)


        // Other methods

        .def("activate", &PopStruct1::set_active)
        .def("reset", &PopStruct1::reset)
        .def("clear", &PopStruct1::clear);




    // ProjStruct0
    nanobind::class_<ProjStruct0>(m, "proj0_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct0::_transmission)
        .def_rw("_axon_transmission", &ProjStruct0::_axon_transmission)
        .def_rw("_update", &ProjStruct0::_update)
        .def_rw("_update_period", &ProjStruct0::_update_period)
        .def_rw("_update_offset", &ProjStruct0::_update_offset)
        .def_rw("_plasticity", &ProjStruct0::_plasticity)

        // Connectivity

        .def("init_from_lil", &ProjStruct0::init_from_lil)
        .def("post_rank", &ProjStruct0::get_post_rank)
        .def("dendrite_size", &ProjStruct0::dendrite_size)
        .def("nb_dendrites", &ProjStruct0::nb_dendrites)
        .def("pre_ranks", &ProjStruct0::get_pre_ranks)
        .def("pre_rank", &ProjStruct0::get_dendrite_pre_rank)
        .def("nb_synapses", &ProjStruct0::nb_synapses)

        // Methods


        // Attributes

        // local attributes
        .def("get_local_attribute_all_double", &ProjStruct0::get_local_attribute_all_double)
        .def("get_local_attribute_row_double", &ProjStruct0::get_local_attribute_row_double)
        .def("get_local_attribute_double", &ProjStruct0::get_local_attribute_double)

        .def("set_local_attribute_all_double", &ProjStruct0::set_local_attribute_all_double)
        .def("set_local_attribute_row_double", &ProjStruct0::set_local_attribute_row_double)
        .def("set_local_attribute_double", &ProjStruct0::set_local_attribute_double)

        // semiglobal attributes
        .def("get_semiglobal_attribute_all_bool", &ProjStruct0::get_semiglobal_attribute_all_bool)
        .def("get_semiglobal_attribute_bool", &ProjStruct0::get_semiglobal_attribute_bool)

        .def("set_semiglobal_attribute_all_bool", &ProjStruct0::set_semiglobal_attribute_all_bool)
        .def("set_semiglobal_attribute_bool", &ProjStruct0::set_semiglobal_attribute_bool)

        // global attributes
        .def("get_global_attribute_double", &ProjStruct0::get_global_attribute_double)
        .def("set_global_attribute_double", &ProjStruct0::set_global_attribute_double)


        // Other methods

        .def("size_in_bytes", &ProjStruct0::size_in_bytes)
        .def("clear", &ProjStruct0::clear);


    // ProjStruct1
    nanobind::class_<ProjStruct1>(m, "proj1_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct1::_transmission)
        .def_rw("_axon_transmission", &ProjStruct1::_axon_transmission)
        .def_rw("_update", &ProjStruct1::_update)
        .def_rw("_update_period", &ProjStruct1::_update_period)
        .def_rw("_update_offset", &ProjStruct1::_update_offset)
        .def_rw("_plasticity", &ProjStruct1::_plasticity)

        // Connectivity

        .def("init_from_lil", &ProjStruct1::init_from_lil)
        .def("post_rank", &ProjStruct1::get_post_rank)
        .def("dendrite_size", &ProjStruct1::dendrite_size)
        .def("nb_dendrites", &ProjStruct1::nb_dendrites)
        .def("pre_ranks", &ProjStruct1::get_pre_ranks)
        .def("pre_rank", &ProjStruct1::get_dendrite_pre_rank)
        .def("nb_synapses", &ProjStruct1::nb_synapses)

        // Methods


        // Attributes

        // local attributes
        .def("get_local_attribute_all_double", &ProjStruct1::get_local_attribute_all_double)
        .def("get_local_attribute_row_double", &ProjStruct1::get_local_attribute_row_double)
        .def("get_local_attribute_double", &ProjStruct1::get_local_attribute_double)

        .def("set_local_attribute_all_double", &ProjStruct1::set_local_attribute_all_double)
        .def("set_local_attribute_row_double", &ProjStruct1::set_local_attribute_row_double)
        .def("set_local_attribute_double", &ProjStruct1::set_local_attribute_double)

        // semiglobal attributes
        .def("get_semiglobal_attribute_all_bool", &ProjStruct1::get_semiglobal_attribute_all_bool)
        .def("get_semiglobal_attribute_bool", &ProjStruct1::get_semiglobal_attribute_bool)

        .def("set_semiglobal_attribute_all_bool", &ProjStruct1::set_semiglobal_attribute_all_bool)
        .def("set_semiglobal_attribute_bool", &ProjStruct1::set_semiglobal_attribute_bool)

        // global attributes
        .def("get_global_attribute_double", &ProjStruct1::get_global_attribute_double)
        .def("set_global_attribute_double", &ProjStruct1::set_global_attribute_double)


        // Other methods

        .def("size_in_bytes", &ProjStruct1::size_in_bytes)
        .def("clear", &ProjStruct1::clear);




    // Monitor for Population 0
    nanobind::class_<PopRecorder0>(m, "PopRecorder0_wrapper")
        // Record flag
		.def_rw("record_x", &PopRecorder0::record_x)
		.def_rw("record_r", &PopRecorder0::record_r)
		.def_rw("record_ex", &PopRecorder0::record_ex)
		.def_rw("record_inp", &PopRecorder0::record_inp)


        // Target container
		.def_rw("x", &PopRecorder0::x)
		.def_rw("r", &PopRecorder0::r)
		.def_rw("ex", &PopRecorder0::ex)
		.def_rw("inp", &PopRecorder0::inp)


        // Clear container
		.def("clear_x", &PopRecorder0::clear_x)
		.def("clear_r", &PopRecorder0::clear_r)
		.def("clear_ex", &PopRecorder0::clear_ex)
		.def("clear_inp", &PopRecorder0::clear_inp)


        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &PopRecorder0::clear)
        .def("size_in_bytes", &PopRecorder0::size_in_bytes);


    // Monitor for Population 1
    nanobind::class_<PopRecorder1>(m, "PopRecorder1_wrapper")
        // Record flag
		.def_rw("record_r", &PopRecorder1::record_r)


        // Target container
		.def_rw("r", &PopRecorder1::r)


        // Clear container
		.def("clear_r", &PopRecorder1::clear_r)


        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &PopRecorder1::clear)
        .def("size_in_bytes", &PopRecorder1::size_in_bytes);




    // Monitor for Projection 0
    nanobind::class_<ProjRecorder0>(m, "ProjRecorder0_wrapper")
        // Record flag
		.def_rw("record_w", &ProjRecorder0::record_w)


        // Target container
		.def_rw("w", &ProjRecorder0::w)


        // Clear container
		.def("clear_w", &ProjRecorder0::clear_w)


        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &ProjRecorder0::clear)
        .def("size_in_bytes", &ProjRecorder0::size_in_bytes);


    // Monitor for Projection 1
    nanobind::class_<ProjRecorder1>(m, "ProjRecorder1_wrapper")
        // Record flag
		.def_rw("record_w", &ProjRecorder1::record_w)


        // Target container
		.def_rw("w", &ProjRecorder1::w)


        // Clear container
		.def("clear_w", &ProjRecorder1::clear_w)


        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &ProjRecorder1::clear)
        .def("size_in_bytes", &ProjRecorder1::size_in_bytes);




}
