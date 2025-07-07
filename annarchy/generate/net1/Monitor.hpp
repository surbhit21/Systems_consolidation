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
        this->x = std::vector< std::vector< double > >();
        this->record_x = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->ex = std::vector< std::vector< double > >();
        this->record_ex = false; 
        this->inp = std::vector< std::vector< double > >();
        this->record_inp = false; 

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

        if(this->record_x && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record 'x' for pop0." << std::endl;
        #endif
            if(!this->partial) {
                this->x.push_back(pop0->x);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0->x[this->ranks[i]]);
                }
                this->x.push_back(tmp);
            }
        }
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
        if(this->record_ex && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record 'ex' for pop0." << std::endl;
        #endif
            if(!this->partial) {
                this->ex.push_back(pop0->ex);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0->ex[this->ranks[i]]);
                }
                this->ex.push_back(tmp);
            }
        }
        if(this->record_inp && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record 'inp' for pop0." << std::endl;
        #endif
            if(!this->partial) {
                this->inp.push_back(pop0->inp);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0->inp[this->ranks[i]]);
                }
                this->inp.push_back(tmp);
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
        
        // local variable x
        size_in_bytes += sizeof(std::vector<double>) * x.capacity();
        for(auto it=x.begin(); it!= x.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable ex
        size_in_bytes += sizeof(std::vector<double>) * ex.capacity();
        for(auto it=ex.begin(); it!= ex.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable inp
        size_in_bytes += sizeof(std::vector<double>) * inp.capacity();
        for(auto it=inp.begin(); it!= inp.end(); it++) {
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
    
    void clear_x() {
        for(auto it = this->x.begin(); it != this->x.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->x.clear();
    }
    
    void clear_r() {
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    }
    
    void clear_ex() {
        for(auto it = this->ex.begin(); it != this->ex.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->ex.clear();
    }
    
    void clear_inp() {
        for(auto it = this->inp.begin(); it != this->inp.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->inp.clear();
    }
    

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::clear() - this = " << this << std::endl;
    #endif
		this->clear__sum_exc();
		this->clear_x();
		this->clear_r();
		this->clear_ex();
		this->clear_inp();


        removeRecorder(this);
    }



    // Local variable _sum_exc
    std::vector< std::vector< double > > _sum_exc ;
    bool record__sum_exc ; 
    // Local variable x
    std::vector< std::vector< double > > x ;
    bool record_x ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable ex
    std::vector< std::vector< double > > ex ;
    bool record_ex ; 
    // Local variable inp
    std::vector< std::vector< double > > inp ;
    bool record_inp ; 
};

class PopRecorder1 : public Monitor
{
protected:
    int _id;

public:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") instantiated." << std::endl;
    #endif

        this->_sum_exc = std::vector< std::vector< double > >();
        this->record__sum_exc = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 

        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif
    }

    ~PopRecorder1() {
    #ifdef _DEBUG
        std::cout << "PopRecorder1::~PopRecorder1() - this = " << this << std::endl;
    #endif
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder1::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record 'r' for pop1." << std::endl;
        #endif
            if(!this->partial) {
                this->r.push_back(pop1->r);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1->r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    Record '_sum_exc' for pop1." << std::endl;
        #endif
            if(!this->partial) {
                this->_sum_exc.push_back(pop1->_sum_exc);
            } else {
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1->_sum_exc[this->ranks[i]]);
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
        std::cout << "PopRecorder1::clear() - this = " << this << std::endl;
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

class ProjRecorder0 : public Monitor
{
protected:

    int _id;
    std::vector <int> indices;

public:

    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0->get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        // initialize container

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif

    };

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0->get_matrix_variable_row<double>(proj0->w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;

        
        // local variable w
        size_in_bytes += sizeof(std::vector<std::vector<double>>) * w.capacity();
        for (auto it=w.begin(); it!= w.end(); it++) {
            for (auto it2=it->begin(); it2!= it->end(); it2++) {
                size_in_bytes += it2->capacity() * sizeof(double);
            }
        }
        

        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor0::clear()." << std::endl;
    #endif
	this->clear_w();

    }


    void clear_w() {
        for (auto it=w.begin(); it!= w.end(); it++) {
            for (auto it2=it->begin(); it2!= it->end(); it2++) {
                it2->clear();
                it2->shrink_to_fit();
            }
        }
        w.clear();
    }



    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

class ProjRecorder1 : public Monitor
{
protected:

    int _id;
    std::vector <int> indices;

public:

    ProjRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj1->get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        // initialize container

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif

    };

    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj1->get_matrix_variable_row<double>(proj1->w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;

        
        // local variable w
        size_in_bytes += sizeof(std::vector<std::vector<double>>) * w.capacity();
        for (auto it=w.begin(); it!= w.end(); it++) {
            for (auto it2=it->begin(); it2!= it->end(); it2++) {
                size_in_bytes += it2->capacity() * sizeof(double);
            }
        }
        

        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor1::clear()." << std::endl;
    #endif
	this->clear_w();

    }


    void clear_w() {
        for (auto it=w.begin(); it!= w.end(); it++) {
            for (auto it2=it->begin(); it2!= it->end(); it2++) {
                it2->clear();
                it2->shrink_to_fit();
            }
        }
        w.clear();
    }



    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

