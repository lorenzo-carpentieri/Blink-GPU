#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <typeinfo>
#include <cstdlib>


namespace common {
    namespace logger {

        
        enum class LogLevel {
            Debug,
            Info
        };

        LogLevel parseLogLevel(const std::string& level)
        {
            if (level == "DEBUG")
                return LogLevel::Debug;
            return LogLevel::Info;
        }

        void logMessage(const std::string& message, LogLevel type)
        {
            const char* env = std::getenv("LOG_INFO");
            if (!env) return;

            std::string envLevel(env);
            std::transform(envLevel.begin(), envLevel.end(),
                        envLevel.begin(), ::toupper);

            LogLevel currentLevel = parseLogLevel(envLevel);

            if (currentLevel == type) {
                std::cerr << "[" << envLevel << "] " << message << std::endl;
            }
        }


        namespace fs = std::filesystem;
        // Utility struct to represent a single field in the CSV log
        // Used to build dynamic CSV rows with extra fields if needed
        struct CsvField {
            std::string key;
            std::string value;

            template<typename T>
            CsvField(std::string k, T&& v)
                    : key(std::move(k))
                    , value(to_string(std::forward<T>(v))) {}

                private:
                    static std::string to_string(const std::string& v) { return v; }
                    static std::string to_string(const char* v) { return v; }

                    template<typename T>
                    static std::string to_string(T v) {
                        std::ostringstream oss;
                        oss << v;
                        return oss.str();
                    }
                };

        class Logger {
        private:
        
            std::string output_file_path;
            std::string library_name;
            std::string collective_name;
            std::string gpu_mode = "gpu";
            
            int run_id;
            int mpi_rank;

            // Nuove variabili per informazioni sui nodi
            std::string hostname;
            int node_id;
            int total_nodes;
            bool is_multi_node;

            
            std::string get_hostname_internal() const {
                char hostname_buf[256];
                if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0) {
                    return std::string(hostname_buf);
                }
                return "unknown";
            }
            
            void initialize_node_info() {
                logMessage("Initialze node info", LogLevel::Info);
                // Ottieni hostname
                hostname = get_hostname_internal();

                // Ottieni informazioni MPI sui nodi
                int rank, size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                mpi_rank = rank;
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                
                // Raccogli tutti gli hostname dai vari rank
                const int max_hostname_len = 256;
                char local_hostname[max_hostname_len];
                strncpy(local_hostname, hostname.c_str(), max_hostname_len - 1);
                local_hostname[max_hostname_len - 1] = '\0';
                
                // Buffer per raccogliere tutti gli hostname
                std::vector<char> all_hostnames(size * max_hostname_len);
                
                // Gather di tutti gli hostname al rank 0
                MPI_Allgather(local_hostname, max_hostname_len, MPI_CHAR,
                            all_hostnames.data(), max_hostname_len, MPI_CHAR,
                            MPI_COMM_WORLD);
                
                // Analizza gli hostname per determinare nodi unici
                std::unordered_set<std::string> unique_hosts;
                std::vector<std::string> host_list;
                
                for (int i = 0; i < size; i++) {
                    std::string host(all_hostnames.data() + i * max_hostname_len);
                    host_list.push_back(host);
                    unique_hosts.insert(host);
                }
                
                // Determina total_nodes
                total_nodes = static_cast<int>(unique_hosts.size());
                
                // Determina is_multi_node
                is_multi_node = (total_nodes > 1);
                
                // Determina node_id (assegna un ID progressivo basato sull'ordine degli hostname unici)
                std::vector<std::string> sorted_hosts(unique_hosts.begin(), unique_hosts.end());
                std::sort(sorted_hosts.begin(), sorted_hosts.end());
                
                node_id = 0;
                for (int i = 0; i < sorted_hosts.size(); i++) {
                    if (sorted_hosts[i] == hostname) {
                        node_id = i;
                        break;
                    }
                }
                
                // Debug output (solo dal rank 0)
                
                if (rank == 0) {
                    std::ostringstream oss;

                    oss << "Total nodes: " << total_nodes 
                            << ", Multi-node: " 
                            << (is_multi_node ? "yes" : "no") << std::endl;
                    oss << "Unique hostnames: ";
                    for (const auto& host : sorted_hosts) {
                        oss << host << " ";
                    }
                    oss << std::endl;
                    
                    logMessage(oss.str(), LogLevel::Info);
                }
            }

        
         
            
            void ensure_directory_exists() const {
                fs::path path(output_file_path);
                // Get the parent directory of the file
                fs::path dir = path.parent_path();

                if (!dir.empty() && !fs::exists(dir)) {
                    // Create directories recursively
                    try {
                        std::ostringstream oss;
                        fs::create_directories(dir); // returns true/false
                        oss << "Directory exists or created: " << dir << "\n";
                        logMessage(oss.str(), LogLevel::Info);
                    } catch (const std::exception& e) {
                        std::ostringstream oss;
                        oss << "Error creating directory: " << dir << " -> " << e.what() << "\n";
                        logMessage(oss.str(), LogLevel::Info);
                    }
                }
            }
            
            bool file_exists(const std::string& filename) const {
                return fs::exists(filename);
            }

            void write_header(std::ofstream& file, const std::vector<CsvField>& extras = {}) const {
                file << "library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,global_rank,local_rank,hostname,node_id,total_nodes,is_multi_node,run_id,gpu_mode,test_passed,chain_size,total_time_ms,time_ms_1coll,total_device_energy_mj,device_energy_mj_1coll,total_host_energy_mj,host_energy_mj_1coll,goodput_Gb_per_s";
                std::cout << "library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,global_rank,local_rank,hostname,node_id,total_nodes,is_multi_node,run_id,gpu_mode,test_passed,chain_size,total_time_ms,time_ms_1coll,total_device_energy_mj,device_energy_mj_1coll,total_host_energy_mj,host_energy_mj_1coll,goodput_Gb_per_s";
 
                for (const auto& f : extras){
                    file << "," << f.key;
                    std::cout << "," << f.key;
                }
                file << "\n";
                std::cout << "\n";

            }

        public:
            Logger(const std::string& output_file_path, const std::string& library_name, const std::string& collective_name, const std::string& gpu_mode = "composite")
                : output_file_path(output_file_path), library_name(library_name), collective_name(collective_name), gpu_mode(gpu_mode) {
                ensure_directory_exists();
                initialize_node_info();
            }
            
             // T define the data type used for the collective operation
            template <typename T>
            struct ProfilingInfo {
                double time_ms; // time to solution of the rank to execute the specified collective in ms
                size_t message_size_bytes;
                double device_energy_mj; // in mj 
                double host_energy_mj; // in mj
                int global_rank;
                int local_rank;
                int num_ranks;
                int run_id; // index of the current run
                bool test_passed; // check the results of the collective operation
                int chain_size; // check the results of the collective operation
                std::string gpu_mode; // composite or flat: only for intel GPUs
            };

            // The log interface take as input ProfilingInfo but we can also add other fields to the csv as ExtraFields...
            template<typename T, typename... ExtraFields>
            void log_result(const ProfilingInfo<T>& info, ExtraFields&&... extras){
                /* 
                   For each message size we execute multiple times the same collective "c" so that we have a chain like this c_1, c_2, ... c_m, where m=chain_size.
                   The ProfilingInfo struct contains the time in ms and energy in mJ for the execution of the whole chain c_1, c_2, ... c_m,
                   To measure the time and energy of a single collective we devide time_ms and {device,host}_energy_mj with chain_size.
                */
               std::vector<CsvField> extra_fields{ std::forward<ExtraFields>(extras)... };

                double time_ms_1coll = info.time_ms / info.chain_size; 
                double device_energy_mj_1coll = info.device_energy_mj / info.chain_size; 
                double host_energy_mj_1coll = info.host_energy_mj / info.chain_size; 
                double data_Gb = static_cast<double>(info.message_size_bytes) / 1.25e+8;
                double goodput_Gb_per_s = data_Gb / (time_ms_1coll / 1000); // Gigabit per seconds 

                
                std::string filename = output_file_path;
                
                // Only rank 0 write the csv header 
                int needs_header = 0;
                if (info.global_rank == 0) {
                    bool is_new_file = !file_exists(filename);
                    needs_header = is_new_file ? 1 : 0;
                }

                MPI_Bcast(&needs_header, 1, MPI_INT, 0, MPI_COMM_WORLD);
                
                // Solo il rank 0 scrive l'header se necessario
                if (needs_header && info.global_rank == 0) {
                    std::ofstream header_file(filename, std::ios::app);
                    if (header_file.is_open()) {
                        write_header(header_file, extra_fields);
                        header_file.close();
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                
                for (int r = 0; r < info.num_ranks; r++) {
                    if (r == info.global_rank) {
                        std::ofstream file(filename, std::ios::app);
                        if (!file.is_open()) {
                            std::cerr << "Warning: Could not open log file: " << filename << std::endl;
                            return;
                        }
                    
                        // Adde energy for each rank, then add also another line for avg time and energy for all the ranks
                        // Scrivi la riga di dati
                        file << library_name << ","
                            << collective_name << ","
                            << typeid(T).name() << ","
                            << info.message_size_bytes << ","
                            << info.message_size_bytes /  sizeof(T) << ","
                            << info.num_ranks << ","
                            << info.global_rank << "," // can be 0 to num_ranks or aggregate
                            << info.local_rank << "," // can be 0 to num_ranks or aggregate
                            << hostname << ","
                            << node_id << ","
                            << total_nodes << ","
                            << (is_multi_node ? "true" : "false") << ","
                            << info.run_id << "," 
                            << gpu_mode << "," // Only for Intel GPUs
                            << (info.test_passed ? "true" : "false") << ","
                            << info.chain_size << ","
                            << std::fixed << std::setprecision(6) << info.time_ms  << ","
                            << std::fixed << std::setprecision(6) << time_ms_1coll  << ","
                            << std::fixed << std::setprecision(3) << info.device_energy_mj  << ","
                            << std::fixed << std::setprecision(3) << device_energy_mj_1coll  << ","
                            << std::fixed << std::setprecision(3) << info.host_energy_mj  << ","
                            << std::fixed << std::setprecision(3) << host_energy_mj_1coll  << ","
                            << std::fixed << std::setprecision(6) << goodput_Gb_per_s;
                            
                            // Write extra fields if any
                            for (const auto& f : extra_fields)
                                file << "," << f.value;
                            
                            file << "\n";
                            
                        file.close();


                        /*################### Print the results also in the std::out ###################*/
                        std::cout << library_name << ","
                            << collective_name << ","
                            << typeid(T).name() << ","
                            << info.message_size_bytes << ","
                            << info.message_size_bytes /  sizeof(T) << ","
                            << info.num_ranks << ","
                            << info.global_rank << "," // can be 0 to num_ranks or aggregate
                            << info.local_rank << "," // can be 0 to num_ranks or aggregate
                            << hostname << ","
                            << node_id << ","
                            << total_nodes << ","
                            << (is_multi_node ? "true" : "false") << ","
                            << info.run_id << "," 
                            << gpu_mode << "," // Only for Intel GPUs
                            << (info.test_passed ? "true" : "false") << ","
                            << info.chain_size << ","
                            << std::fixed << std::setprecision(6) << info.time_ms  << ","
                            << std::fixed << std::setprecision(6) << time_ms_1coll  << ","
                            << std::fixed << std::setprecision(3) << info.device_energy_mj  << ","
                            << std::fixed << std::setprecision(3) << device_energy_mj_1coll  << ","
                            << std::fixed << std::setprecision(3) << info.host_energy_mj  << ","
                            << std::fixed << std::setprecision(3) << host_energy_mj_1coll  << ","
                            << std::fixed << std::setprecision(6) << goodput_Gb_per_s;
                            
                            // Write extra fields if any
                            for (const auto& f : extra_fields)
                                std::cout << "," << f.value;
                            
                            std::cout << "\n";
                            
                    }
                    
                    MPI_Barrier(MPI_COMM_WORLD);

                }
            }
          
            
            
            // Metodi getter per le nuove informazioni sui nodi
            const std::string& get_hostname() const { return hostname; }
            int get_node_id() const { return node_id; }
            int get_total_nodes() const { return total_nodes; }
            bool get_is_multi_node() const { return is_multi_node; }
            
         
            
            static void print_usage() {
                std::cout << "\nLogger Usage:" << std::endl;
                std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
                std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
                std::cout << "\nOutput format: CSV files with columns:" << std::endl;
                std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, gpu_mode, test_passed, time_ms" << std::endl;
            }


        };
    } // end namespace logger
} // end namespace common