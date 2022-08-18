#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
	std::chrono::time_point<std::chrono::system_clock> start;
	std::chrono::time_point<std::chrono::system_clock> finish;
	std::string function_name;
	std::chrono::duration<double> time;
public:
	Timer(std::string&& name) :
		time()
	{
		function_name = std::move(name);
	}

	void resume()
	{
    	start = std::chrono::high_resolution_clock::now();
	}

	void pause()
	{
	    finish = std::chrono::high_resolution_clock::now();
	    auto spent = finish - start;
	    time += spent;
	}

	void result()
	{
		using milli = std::chrono::milliseconds;
	    std::cout << function_name << "() took " << 
	    std::chrono::duration_cast<milli>(time).count() << " milliseconds\n";
	}
};