#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cache.h"
#include <memory>

namespace {
using finemoe::ExpertCache;

class ReleaseGIL {
    PyThreadState* state_ = PyEval_SaveThread();
public:
    ~ReleaseGIL() { PyEval_RestoreThread(state_); }
};

struct Handle {
    std::unique_ptr<ExpertCache> cache;
    PyObject* owners = nullptr;
    PyObject* graphs = nullptr;
    ~Handle() {
        { ReleaseGIL release; cache.reset(); }
        Py_XDECREF(graphs);
        Py_XDECREF(owners);
    }
};

constexpr const char* capsule_name = "finemoe.ExpertCache";

struct PythonError {};

template <typename F> PyObject* api(F&& function) {
    try { return function(); }
    catch (const PythonError&) { return nullptr; }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::invalid_argument& error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    return nullptr;
}

Handle& handle(PyObject* capsule) {
    auto* result = static_cast<Handle*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!result) throw PythonError{};
    return *result;
}

void destroy(PyObject* capsule) { delete static_cast<Handle*>(PyCapsule_GetPointer(capsule, capsule_name)); }

class Sequence {
    PyObject* value_;
public:
    explicit Sequence(PyObject* value) : value_(PySequence_Fast(value, "expected a sequence")) {
        if (!value_) throw PythonError{};
    }
    ~Sequence() { Py_DECREF(value_); }
    Py_ssize_t size() const { return PySequence_Fast_GET_SIZE(value_); }
    PyObject* operator[](Py_ssize_t index) const { return PySequence_Fast_GET_ITEM(value_, index); }
};

unsigned long long integer(PyObject* value) {
    auto result = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) throw PythonError{};
    return result;
}

int index(PyObject* value, std::size_t size) {
    auto result = integer(value);
    if (result >= size) throw std::invalid_argument("cache index out of range");
    return static_cast<int>(result);
}

void* pointer(unsigned long long value) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value)); }

PyObject* create(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        const char* library;
        int device;
        unsigned long long stream, bytes;
        PyObject *sources, *destinations, *owners;
        if (!PyArg_ParseTuple(args, "siKKOOO", &library, &device, &stream, &bytes,
                              &sources, &destinations, &owners)) return nullptr;
        Sequence source_list(sources), destination_list(destinations);
        std::vector<finemoe::Expert> experts;
        std::vector<void*> slots;
        for (Py_ssize_t i = 0; i < source_list.size(); ++i) {
            Sequence entry(source_list[i]);
            if (entry.size() != 2) throw std::invalid_argument("expected layer and host address");
            experts.push_back(finemoe::Expert{index(entry[0], INT_MAX), pointer(integer(entry[1]))});
        }
        for (Py_ssize_t i = 0; i < destination_list.size(); ++i)
            slots.push_back(pointer(integer(destination_list[i])));
        auto result = std::make_unique<Handle>();
        {
            ReleaseGIL release;
            result->cache = std::make_unique<ExpertCache>(library, device, pointer(stream), bytes,
                                                         std::move(experts), slots);
        }
        Py_INCREF(owners);
        result->owners = owners;
        PyObject* capsule = PyCapsule_New(result.get(), capsule_name, destroy);
        if (capsule) result.release();
        return capsule;
    });
}

PyObject* contains(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *value;
        if (!PyArg_ParseTuple(args, "OO", &capsule, &value)) return nullptr;
        auto& cache = *handle(capsule).cache;
        int expert = index(value, cache.size());
        bool result;
        { ReleaseGIL release; result = cache.contains(expert); }
        return PyBool_FromLong(result);
    });
}

PyObject* prefetch(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *value;
        if (!PyArg_ParseTuple(args, "OO", &capsule, &value)) return nullptr;
        auto& cache = *handle(capsule).cache;
        int expert = index(value, cache.size());
        bool result;
        { ReleaseGIL release; result = cache.prefetch(expert); }
        return PyBool_FromLong(result);
    });
}

PyObject* expire(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject* capsule;
        int layer, all;
        if (!PyArg_ParseTuple(args, "Oip", &capsule, &layer, &all)) return nullptr;
        auto& cache = *handle(capsule).cache;
        { ReleaseGIL release; cache.expire(layer, all); }
        Py_RETURN_NONE;
    });
}

std::pair<std::vector<int>, std::vector<double>> read_probabilities(
        ExpertCache& cache, PyObject* positions, PyObject* values) {
    Sequence indices(positions);
    std::vector<int> parsed;
    parsed.reserve(indices.size());
    for (Py_ssize_t i = 0; i < indices.size(); ++i) parsed.push_back(index(indices[i], cache.size()));
    Py_buffer buffer;
    if (PyObject_GetBuffer(values, &buffer, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS)) throw PythonError{};
    std::vector<double> probabilities;
    try {
        if (buffer.ndim != 1 || buffer.itemsize != sizeof(double) ||
            std::string(buffer.format) != "d" || buffer.len / buffer.itemsize != indices.size())
            throw std::invalid_argument("expected one float64 probability per expert");
        auto* data = static_cast<const double*>(buffer.buf);
        probabilities.assign(data, data + indices.size());
    } catch (...) { PyBuffer_Release(&buffer); throw; }
    PyBuffer_Release(&buffer);
    return {std::move(parsed), std::move(probabilities)};
}

PyObject* probabilities(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *positions, *values;
        if (!PyArg_ParseTuple(args, "OOO", &capsule, &positions, &values)) return nullptr;
        auto& cache = *handle(capsule).cache;
        auto parsed = read_probabilities(cache, positions, values);
        { ReleaseGIL release; cache.probabilities(parsed.first, parsed.second); }
        Py_RETURN_NONE;
    });
}

PyObject* plan(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *positions, *values, *order;
        int layer;
        if (!PyArg_ParseTuple(args, "OiOOO", &capsule, &layer, &positions, &values, &order)) return nullptr;
        auto& cache = *handle(capsule).cache;
        auto parsed = read_probabilities(cache, positions, values);
        Sequence ranking(order);
        std::vector<int> ranked;
        ranked.reserve(ranking.size());
        for (Py_ssize_t i = 0; i < ranking.size(); ++i) ranked.push_back(index(ranking[i], parsed.first.size()));
        { ReleaseGIL release; cache.plan(layer, parsed.first, parsed.second, ranked); }
        Py_RETURN_NONE;
    });
}

PyObject* advance(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject* capsule;
        int layer;
        if (!PyArg_ParseTuple(args, "Oi", &capsule, &layer)) return nullptr;
        auto& cache = *handle(capsule).cache;
        { ReleaseGIL release; cache.advance(layer); }
        Py_RETURN_NONE;
    });
}

PyObject* reset_probabilities(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject* capsule;
        double value;
        if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return nullptr;
        auto& cache = *handle(capsule).cache;
        { ReleaseGIL release; cache.reset_probabilities(value); }
        Py_RETURN_NONE;
    });
}

PyObject* acquire(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *value, *next;
        unsigned long long stream;
        if (!PyArg_ParseTuple(args, "OOOK", &capsule, &value, &next, &stream)) return nullptr;
        auto& cache = *handle(capsule).cache;
        int expert = index(value, cache.size()), following = next == Py_None ? -1 : index(next, cache.size());
        int result;
        { ReleaseGIL release; result = cache.acquire(expert, following, pointer(stream)); }
        return PyLong_FromLong(result);
    });
}

PyObject* release(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *value;
        unsigned long long stream;
        if (!PyArg_ParseTuple(args, "OOK", &capsule, &value, &stream)) return nullptr;
        auto& cache = *handle(capsule).cache;
        int slot = index(value, cache.capacity());
        { ReleaseGIL release; cache.release(slot, pointer(stream)); }
        Py_RETURN_NONE;
    });
}

PyObject* configure(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *graphs, *owners;
        unsigned long long input, weight, accumulator, bytes;
        if (!PyArg_ParseTuple(args, "OOKKKKO", &capsule, &graphs, &input, &weight, &accumulator,
                              &bytes, &owners)) return nullptr;
        auto& target = handle(capsule);
        Sequence values(graphs);
        if (static_cast<std::size_t>(values.size()) != target.cache->capacity())
            throw std::invalid_argument("expected one graph per slot");
        std::vector<void*> addresses;
        for (Py_ssize_t i = 0; i < values.size(); ++i) addresses.push_back(pointer(integer(values[i])));
        finemoe::Workspace workspace{pointer(input), pointer(weight), pointer(accumulator), bytes};
        { ReleaseGIL release; target.cache->configure(addresses, workspace); }
        Py_INCREF(owners);
        Py_XSETREF(target.graphs, owners);
        Py_RETURN_NONE;
    });
}

PyObject* dispatch(PyObject*, PyObject* args) {
    return api([&]() -> PyObject* {
        PyObject *capsule, *ranked;
        unsigned long long hidden, weights, stride, shared, stream;
        if (!PyArg_ParseTuple(args, "OOKKKKK", &capsule, &ranked, &hidden, &weights, &stride,
                              &shared, &stream)) return nullptr;
        auto& cache = *handle(capsule).cache;
        Sequence ranks(ranked);
        std::vector<std::pair<int, int>> experts;
        for (Py_ssize_t i = 0; i < ranks.size(); ++i) {
            Sequence entry(ranks[i]);
            if (entry.size() != 2) throw std::invalid_argument("expected rank and expert index");
            experts.emplace_back(index(entry[0], ranks.size()), index(entry[1], cache.size()));
        }
        { ReleaseGIL release; cache.dispatch(experts, pointer(hidden), pointer(weights), stride,
                                            pointer(shared), pointer(stream)); }
        Py_RETURN_NONE;
    });
}

PyObject* stats(PyObject*, PyObject* capsule) {
    return api([&]() -> PyObject* {
        auto& cache = *handle(capsule).cache;
        std::pair<std::uint64_t, std::uint64_t> result;
        { ReleaseGIL release; result = cache.stats(); }
        return Py_BuildValue("KK", static_cast<unsigned long long>(result.first),
                            static_cast<unsigned long long>(result.second));
    });
}

PyObject* clear(PyObject*, PyObject* capsule) {
    return api([&]() -> PyObject* {
        auto& cache = *handle(capsule).cache;
        { ReleaseGIL release; cache.clear(); }
        Py_RETURN_NONE;
    });
}

PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, nullptr},
    {"contains", contains, METH_VARARGS, nullptr},
    {"prefetch", prefetch, METH_VARARGS, nullptr},
    {"expire", expire, METH_VARARGS, nullptr},
    {"probabilities", probabilities, METH_VARARGS, nullptr},
    {"plan", plan, METH_VARARGS, nullptr},
    {"advance", advance, METH_VARARGS, nullptr},
    {"reset_probabilities", reset_probabilities, METH_VARARGS, nullptr},
    {"acquire", acquire, METH_VARARGS, nullptr},
    {"release", release, METH_VARARGS, nullptr},
    {"configure", configure, METH_VARARGS, nullptr},
    {"dispatch", dispatch, METH_VARARGS, nullptr},
    {"stats", stats, METH_O, nullptr},
    {"clear", clear, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cache", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__cache() { return PyModule_Create(&module); }
