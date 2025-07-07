#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    int _id;

public:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

        this->_sum_exc = std::vector< std::vector< double > >();
        this->record__sum_exc = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 

        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif
    }

    ~PopRecorder0() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::~PopRecorder0() - this = " << this << std::endl;
    #endif
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record 'r' for pop0." << std::endl;
        #endif
            if(!this->partial) {
                this->r.push_back(pop0->r);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0->r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record '_sum_exc' for pop0." << std::endl;
        #endif
            if(!this->partial) {
                this->_sum_exc.push_back(pop0->_sum_exc);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0->_sum_exc[this->ranks[i]]);
                }
                this->_sum_exc.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        return size_in_bytes;
    }


    void clear__sum_exc() {
        for(auto it = this->_sum_exc.begin(); it != this->_sum_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->_sum_exc.clear();
    }
    
    void clear_r() {
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    }
    

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::clear() - this = " << this << std::endl;
    #endif
		this->clear__sum_exc();
		this->clear_r();


        removeRecorder(this);
    }



    // Local variable _sum_exc
    std::vector< std::vector< double > > _sum_exc ;
    bool record__sum_exc ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

