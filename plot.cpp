/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   plot.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jpinyot <marvin@42.fr>                     +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2020/02/14 09:00:03 by jpinyot           #+#    #+#             */
/*   Updated: 2020/02/21 10:56:06 by jpinyot          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "plot.h"

PyObject* Plot::getArray(const vector<double>& array)
{
    npy_intp dims = array.size();
	void* value = (void*)array.data();
	PyObject* retArray = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, value);
    return retArray;
}

bool		Plot::needFloatVal(const string& string)
{
	if (string == "linewidth"){
		return true;
	}
	return false;
}

void	Plot::initialize()
{
	Py_Initialize();
	if(PyArray_API == NULL){
		import_array();
	}
}

void	Plot::plot(const vector<double>& y, const string format)
{
	vector<double> x(y.size());
    for(size_t i=0; i<x.size(); ++i) x.at(i) = i;

    PyObject* xarray = getArray(x);
    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, pystring);

	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);
	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "plot"),
											args);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if(res) Py_DECREF(res);
}

void	Plot::plot(const vector<double>& x, const vector<double>& y, const string format)
{
    PyObject* xarray = getArray(x);
    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, pystring);

	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);
	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "plot"),
											args);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if(res) Py_DECREF(res);
}

void	Plot::named_plot(const string& name, const vector<double>& y, const string& format)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(2);

    PyTuple_SetItem(args, 0, yarray);
    PyTuple_SetItem(args, 1, pystring);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "plot"),
									args,
									kwargs);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
}

void	Plot::named_plot(const string& name, const vector<double>& x, const vector<double>& y, const string& format)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = getArray(x);
    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, pystring);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "plot"),
									args,
									kwargs);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
}

void	Plot::param_plot(const vector<string>& params, const vector<double>& x, const vector<double>& y, const string& format)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

	PyObject* kwargs = PyDict_New();
	for (int i = 1; i < params.size(); i += 2){
		if (needFloatVal(params[i - 1])){
			PyDict_SetItemString(kwargs, params[i - 1].c_str(), PyFloat_FromDouble(stod(params[i])));
		}
		else{
			PyDict_SetItemString(kwargs, params[i - 1].c_str(), PyString_FromString(params[i].c_str()));
		}
	}

    PyObject* xarray = getArray(x);
    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, pystring);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "plot"),
									args,
									kwargs);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
}
void	Plot::param_plot(const vector<string>& params, const vector<double>& y, const string& format)
{
	PyObject* kwargs = PyDict_New();
	for (int i = 1; i < params.size(); i += 2){
		if (needFloatVal(params[i - 1])){
			PyDict_SetItemString(kwargs, params[i - 1].c_str(), PyFloat_FromDouble(stod(params[i])));
		}
		else{
			PyDict_SetItemString(kwargs, params[i - 1].c_str(), PyString_FromString(params[i].c_str()));
		}
	}

	vector<double> x(y.size());
    for(size_t i=0; i<x.size(); ++i) x.at(i) = i;

    PyObject* xarray = getArray(x);
    PyObject* yarray = getArray(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, pystring);

	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "plot"),
									args,
									kwargs);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
}

void	Plot::subplot(long nRows, long nCols, long plotNumber)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(nRows));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(nCols));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(plotNumber));

	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "subplot"),
											args);
    if(!res) throw runtime_error("Call to subplot() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
}

void	Plot::subplot2grid(long nRows, long nCols, long rowId, long colId, long rowSpan, long colSpan)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* shape = PyTuple_New(2);
    PyTuple_SetItem(shape, 0, PyLong_FromLong(nRows));
    PyTuple_SetItem(shape, 1, PyLong_FromLong(nCols));

    PyObject* loc = PyTuple_New(2);
    PyTuple_SetItem(loc, 0, PyLong_FromLong(rowId));
    PyTuple_SetItem(loc, 1, PyLong_FromLong(colId));

    PyObject* args = PyTuple_New(4);
    PyTuple_SetItem(args, 0, shape);
    PyTuple_SetItem(args, 1, loc);
    PyTuple_SetItem(args, 2, PyLong_FromLong(rowSpan));
    PyTuple_SetItem(args, 3, PyLong_FromLong(colSpan));

	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "subplot2grid"),
											args);
    if(!res) throw runtime_error("Call to subplot2grid() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(shape);
    Py_DECREF(loc);
    Py_DECREF(args);
    if (res) Py_DECREF(res);
	
}

void	Plot::subplots_adjust(const vector<string>& keywords)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

	PyObject* kwargs = PyDict_New();
	for (int i = 1; i < keywords.size(); i += 2){
		if (needFloatVal(keywords[i - 1])){
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyFloat_FromDouble(stod(keywords[i])));
		}
		else{
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyString_FromString(keywords[i].c_str()));
		}
	}

    PyObject* args = PyTuple_New(0);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "subplots_adjust"),
									args,
									kwargs);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);
}

void	Plot::axis(const string &axisstr)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* str = PyString_FromString(axisstr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, str);

	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "axis"),
											args);
    if(!res) throw runtime_error("Call to title() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if(res) Py_DECREF(res);
}
void	ylim(double left, double right)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "ylim"),
											args);
    if(!res) throw runtime_error("Call to ylim() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if(res) Py_DECREF(res);
}

void	xlim(double left, double right)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "xlim"),
											args);
    if(!res) throw runtime_error("Call to ylim() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    if(res) Py_DECREF(res);
}

void	Plot::tick_params(const map<string, string>& keywords, const string axis)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

	PyObject* args;
	args = PyTuple_New(1);
	PyTuple_SetItem(args, 0, PyString_FromString(axis.c_str()));
	
	PyObject* kwargs = PyDict_New();
	for (map<string, string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it){
		PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
	}

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "tick_params"),
									args, kwargs);
  
	if (!res) throw runtime_error("Call to tick_params() failed");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
	Py_DECREF(args);
	Py_DECREF(kwargs);
    if(res) Py_DECREF(res);
}

void	Plot::legend(const vector<string>& keywords)
{
	PyObject* kwargs = PyDict_New();
	for (int i = 1; i < keywords.size(); i += 2){
		if (needFloatVal(keywords[i - 1])){
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyFloat_FromDouble(stod(keywords[i])));
		}
		else{
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyString_FromString(keywords[i].c_str()));
		}
	}
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);
    PyObject* empty_tube = PyTuple_New(0);

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "legend"),
									empty_tube,
									kwargs);
    if(!res) throw std::runtime_error("Call to legend() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
	Py_DECREF(kwargs);
	Py_DECREF(empty_tube);
    if(res) Py_DECREF(res);
}

void	Plot::title(const string& titlestr, const vector<string>& keywords)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pytitlestr);

    PyObject* kwargs = PyDict_New();
	for (int i = 1; i < keywords.size(); i += 2){
		if (needFloatVal(keywords[i - 1])){
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyFloat_FromDouble(stod(keywords[i])));
		}
		else{
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyString_FromString(keywords[i].c_str()));
		}
	}

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "title"),
									args,
									kwargs);
    if(!res) throw std::runtime_error("Call to title() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

}

void	Plot::xlabel(const string &str, const vector<string>& keywords)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
	for (int i = 1; i < keywords.size(); i += 2){
		if (needFloatVal(keywords[i - 1])){
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyFloat_FromDouble(stod(keywords[i])));
		}
		else{
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyString_FromString(keywords[i].c_str()));
		}
	}

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "xlabel"),
									args,
									kwargs);
    if(!res) throw std::runtime_error("Call to xlabel() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);
}

void	Plot::ylabel(const string &str, const vector<string>& keywords)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
	for (int i = 1; i < keywords.size(); i += 2){
		if (needFloatVal(keywords[i - 1])){
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyFloat_FromDouble(stod(keywords[i])));
		}
		else{
			PyDict_SetItemString(kwargs, keywords[i - 1].c_str(), PyString_FromString(keywords[i].c_str()));
		}
	}

	PyObject* res = PyObject_Call(	PyObject_GetAttrString(pymod, "ylabel"),
									args,
									kwargs);
    if(!res) throw std::runtime_error("Call to ylabel() failed.");

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);
	
}

void	Plot::tight_layout()
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

    PyObject *res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "tight_layout"),
											PyTuple_New(0));

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    if (res) Py_DECREF(res);
}

void	Plot::set_tight_layout(bool flag)
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);

	PyObject* pyflag = flag ? Py_True : Py_False;
    Py_INCREF(pyflag);

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyflag);
    PyObject *res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "set_tight_layout"),
											args);
	write(1, "$", 1);

	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
	Py_DECREF(args);
    if (res) Py_DECREF(res);
}

void	Plot::show()
{
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* pymod = PyImport_Import(pyplotname);
	PyObject* res = PyObject_CallObject(	PyObject_GetAttrString(pymod, "show"),
											PyTuple_New(0));
	Py_DECREF(pyplotname);
	Py_DECREF(pymod);
    if(res) Py_DECREF(res);
}
