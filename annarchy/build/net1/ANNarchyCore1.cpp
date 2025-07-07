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
		.def_rw("Iext", &PopStruct0::Iext)
		.def_rw("r", &PopStruct0::r)
		.def_rw("_sum_exc", &PopStruct0::_sum_exc)


        // Other methods

        .def("activate", &PopStruct0::set_active)
        .def("reset", &PopStruct0::reset)
        .def("clear", &PopStruct0::clear);






    // Monitor for Population 0
    nanobind::class_<PopRecorder0>(m, "PopRecorder0_wrapper")
        // Record flag
		.def_rw("record_r", &PopRecorder0::record_r)


        // Target container
		.def_rw("r", &PopRecorder0::r)


        // Clear container
		.def("clear_r", &PopRecorder0::clear_r)


        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &PopRecorder0::clear)
        .def("size_in_bytes", &PopRecorder0::size_in_bytes);






}
